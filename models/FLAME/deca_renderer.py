import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import util
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import rasterize_meshes

class Pytorch3dRasterizer(nn.Module):
    """  Borrowed from https://github.com/facebookresearch/pytorch3d
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin':  None,
            'perspective_correct': False,
        }
        # dictionary to object
        class Struct:
            def __init__(self, **entries):
                self.__dict__.update(entries)
        raster_settings = Struct(**raster_settings)
        
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None, h=None, w=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[...,:2] = -fixed_vertices[...,:2]
        raster_settings = self.raster_settings
        if h is None and w is None:
            image_size = raster_settings.image_size
        else:
            image_size = [h, w]
            if h>w:
                fixed_vertices[..., 1] = fixed_vertices[..., 1]*h/w
            else:
                fixed_vertices[..., 0] = fixed_vertices[..., 0]*w/h
            
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone(); attributes = attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:,:,:,0].permute(0,3,1,2)
        pixel_vals = torch.cat([pixel_vals, vismask[:,:,:,0][:,None,:,:]], dim=1)
        return pixel_vals

def keep_vertices_and_update_faces(faces, vertices_to_keep):
    """
    Keep specified vertices in the mesh and update the faces.

    Parameters:
    vertices (torch.Tensor): Tensor of shape (N, 3) representing vertices.
    faces (torch.Tensor): Tensor of shape (F, 3) representing faces, with each value being a vertex index.
    vertices_to_keep (list or torch.Tensor): List or tensor of vertex indices to keep.

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: Updated vertices and faces tensors.
    """
    # Convert vertices_to_keep to a tensor if it's a list
    if isinstance(vertices_to_keep, list):
        vertices_to_keep = torch.tensor(vertices_to_keep, dtype=torch.long)

    # Ensure vertices_to_keep is unique and sorted
    vertices_to_keep = torch.unique(vertices_to_keep)

    # Create a mask for vertices to keep
    mask = torch.zeros(5023, dtype=torch.bool)
    mask[vertices_to_keep] = True


    # Create a mapping from old vertex indices to new ones
    new_vertex_indices = torch.full((5023,), -1, dtype=torch.long)
    new_vertex_indices[mask] = torch.arange(len(vertices_to_keep))

    # Remove faces that reference removed vertices (where mapping is -1)
    valid_faces_mask = (new_vertex_indices[faces] != -1).all(dim=1)
    filtered_faces = faces[valid_faces_mask]

    # Update face indices to new vertex indices
    updated_faces = new_vertex_indices[filtered_faces]

    return updated_faces

class SRenderY(nn.Module):
    def __init__(self, config, obj_filename='assets/FLAME2020/head_template.obj'):
        super(SRenderY, self).__init__()
        self.image_size = config.image_size
        self.config = config

        self.rasterizer = Pytorch3dRasterizer(config.image_size)
        verts, faces, aux = load_obj(obj_filename)
        uvcoords = aux.verts_uvs[None, ...]      # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...] # (N, F, 3)
        faces = faces.verts_idx[None,...]

        # shape colors, for rendering shape overlay
        colors = torch.tensor([180, 180, 180])[None, None, :].repeat(1, faces.max()+1, 1).float()/255.

        colors2 = torch.tensor([180, 180, 180])[None, None, :].repeat(1, faces.max()+1, 1).float()/255.

        import pickle
        flame_masks = pickle.load(
            open('assets/FLAME_masks/FLAME_masks.pkl', 'rb'),
            encoding='latin1')
        self.flame_masks = flame_masks
        if not self.config.render.full_head:


            # self.face_mask = [16, 17, 18, 27, 182, 183, 184, 200, 201, 213, 214, 223, 224, 250, 271, 335, 336, 337, 338, 468, 471, 477, 480, 481, 484, 486, 487, 498, 499, 500, 501, 532, 560, 561, 562, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 589, 590, 591, 592, 605, 622, 623, 624, 625, 626, 627, 628, 629, 630, 667, 668, 669, 670, 671, 672, 673, 674, 679, 680, 681, 682, 683, 688, 691, 692, 693, 694, 695, 696, 697, 702, 703, 704, 705, 706, 707, 708, 713, 714, 715, 723, 724, 725, 728, 729, 730, 731, 732, 734, 735, 738, 739, 759, 764, 765, 766, 767, 774, 784, 785, 797, 802, 807, 808, 809, 814, 815, 816, 821, 822, 823, 824, 825, 826, 827, 828, 829, 864, 865, 866, 867, 868, 877, 878, 879, 880, 881, 882, 883, 884, 885, 893, 894, 896, 897, 898, 899, 902, 903, 904, 905, 906, 907, 908, 909, 918, 919, 920, 921, 922, 923, 924, 926, 927, 928, 929, 933, 934, 939, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 977, 978, 979, 980, 985, 986, 991, 992, 993, 994, 995, 999, 1009, 1012, 1013, 1014, 1015, 1019, 1020, 1021, 1022, 1023, 1033, 1034, 1043, 1044, 1050, 1059, 1060, 1062, 1087, 1088, 1092, 1093, 1096, 1101, 1108, 1113, 1114, 1135, 1144, 1146, 1151, 1152, 1153, 1154, 1155, 1161, 1162, 1163, 1164, 1168, 1169, 1170, 1175, 1176, 1181, 1182, 1183, 1184, 1189, 1190, 1193, 1194, 1195, 1200, 1201, 1202, 1216, 1217, 1218, 1224, 1225, 1226, 1243, 1244, 1292, 1293, 1294, 1329, 1331, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1383, 1384, 1385, 1387, 1388, 1389, 1390, 1391, 1392, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1530, 1531, 1569, 1570, 1571, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1590, 1591, 1593, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1616, 1617, 1618, 1619, 1622, 1623, 1624, 1625, 1626, 1628, 1629, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1657, 1658, 1661, 1662, 1663, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1740, 1743, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1763, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1787, 1788, 1789, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1823, 1824, 1826, 1827, 1836, 1846, 1847, 1848, 1849, 1850, 1853, 1863, 1864, 1865, 1866, 1869, 1871, 1895, 1935, 1936, 1937, 1960, 1961, 1962, 1963, 1982, 1983, 1984, 1985, 1998, 1999, 2000, 2001, 2004, 2009, 2010, 2011, 2012, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2030, 2034, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2125, 2126, 2127, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2160, 2161, 2165, 2166, 2167, 2168, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197, 2198, 2199, 2202, 2203, 2204, 2205, 2206, 2207, 2210, 2211, 2212, 2213, 2214, 2216, 2217, 2220, 2221, 2233, 2238, 2239, 2240, 2241, 2248, 2252, 2256, 2259, 2264, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2297, 2298, 2299, 2300, 2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2324, 2325, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2336, 2337, 2338, 2339, 2340, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2370, 2371, 2372, 2373, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2384, 2385, 2388, 2389, 2391, 2399, 2400, 2401, 2402, 2403, 2404, 2405, 2406, 2407, 2418, 2421, 2422, 2425, 2426, 2427, 2428, 2429, 2430, 2431, 2432, 2433, 2434, 2435, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2455, 2456, 2465, 2466, 2471, 2472, 2473, 2485, 2486, 2487, 2488, 2489, 2490, 2491, 2492, 2493, 2494, 2495, 2496, 2497, 2498, 2499, 2500, 2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2518, 2519, 2520, 2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2532, 2533, 2534, 2535, 2536, 2537, 2538, 2539, 2540, 2541, 2542, 2543, 2544, 2545, 2546, 2547, 2548, 2549, 2550, 2551, 2552, 2553, 2554, 2555, 2556, 2557, 2558, 2559, 2562, 2563, 2564, 2565, 2566, 2567, 2568, 2569, 2570, 2571, 2572, 2573, 2574, 2575, 2576, 2577, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2585, 2586, 2587, 2588, 2589, 2590, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2608, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2666, 2667, 2705, 2706, 2707, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2720, 2721, 2722, 2723, 2726, 2727, 2729, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739, 2740, 2741, 2742, 2743, 2744, 2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2758, 2759, 2760, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769, 2774, 2775, 2778, 2779, 2780, 2784, 2785, 2786, 2787, 2788, 2789, 2790, 2791, 2792, 2793, 2794, 2795, 2796, 2797, 2798, 2799, 2800, 2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809, 2810, 2811, 2812, 2813, 2814, 2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2825, 2826, 2827, 2828, 2829, 2830, 2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2841, 2842, 2843, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2855, 2858, 2863, 2864, 2865, 2866, 2867, 2868, 2869, 2871, 2873, 2874, 2875, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888, 2889, 2890, 2891, 2892, 2894, 2895, 2896, 2897, 2898, 2899, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912, 2913, 2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924, 2925, 2926, 2928, 2929, 2934, 2935, 2936, 2937, 2938, 2939, 2946, 2947, 2948, 2949, 2952, 2953, 2973, 3054, 3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063, 3064, 3068, 3070, 3078, 3079, 3080, 3081, 3082, 3083, 3084, 3085, 3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3098, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125, 3126, 3127, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3143, 3144, 3145, 3146, 3147, 3148, 3149, 3150, 3151, 3152, 3153, 3154, 3155, 3156, 3157, 3158, 3159, 3160, 3161, 3162, 3172, 3173, 3183, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3392, 3393, 3394, 3395, 3396, 3397, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3413, 3414, 3415, 3416, 3417, 3418, 3419, 3420, 3421, 3422, 3423, 3424, 3425, 3426, 3427, 3428, 3429, 3430, 3431, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3450, 3451, 3464, 3465, 3466, 3467, 3468, 3469, 3470, 3471, 3472, 3473, 3474, 3475, 3476, 3477, 3478, 3479, 3480, 3481, 3482, 3483, 3484, 3485, 3486, 3487, 3489, 3490, 3491, 3492, 3493, 3495, 3499, 3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508, 3509, 3511, 3512, 3513, 3515, 3516, 3518, 3520, 3521, 3524, 3526, 3527, 3531, 3533, 3534, 3537, 3538, 3540, 3541, 3542, 3543, 3546, 3547, 3548, 3550, 3551, 3552, 3553, 3555, 3556, 3560, 3561, 3563, 3564, 3571, 3572, 3573, 3575, 3577, 3578, 3579, 3580, 3581, 3582, 3583, 3584, 3585, 3586, 3587, 3588, 3589, 3590, 3591, 3592, 3593, 3594, 3595, 3596, 3597, 3598, 3599, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3610, 3611, 3612, 3613, 3614, 3615, 3617, 3618, 3619, 3620, 3621, 3622, 3623, 3624, 3625, 3626, 3627, 3628, 3629, 3630, 3631, 3632, 3633, 3634, 3635, 3636, 3637, 3639, 3640, 3641, 3642, 3643, 3644, 3645, 3646, 3647, 3648, 3649, 3650, 3651, 3652, 3653, 3654, 3655, 3656, 3657, 3658, 3659, 3660, 3661, 3662, 3663, 3664, 3665, 3666, 3667, 3668, 3669, 3670, 3671, 3672, 3673, 3674, 3675, 3676, 3677, 3678, 3679, 3680, 3681, 3682, 3683, 3684, 3685, 3686, 3687, 3688, 3689, 3690, 3691, 3692, 3693, 3694, 3696, 3699, 3704, 3705, 3706, 3708, 3710, 3711, 3712, 3714, 3715, 3716, 3717, 3718, 3720, 3721, 3722, 3724, 3725, 3726, 3727, 3728, 3729, 3730, 3731, 3732, 3733, 3734, 3735, 3737, 3738, 3739, 3740, 3741, 3742, 3743, 3744, 3745, 3747, 3748, 3749, 3750, 3752, 3753, 3754, 3756, 3757, 3759, 3761, 3762, 3763, 3764, 3766, 3767, 3769, 3770, 3771, 3772, 3773, 3774, 3775, 3776, 3777, 3779, 3780, 3781, 3782, 3784, 3786, 3787, 3788, 3789, 3790, 3791, 3792, 3793, 3794, 3795, 3796, 3797, 3798, 3799, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3808, 3809, 3810, 3811, 3812, 3813, 3814, 3815, 3816, 3817, 3818, 3819, 3820, 3821, 3822, 3823, 3825, 3826, 3827, 3828, 3829, 3830, 3831, 3832, 3833, 3834, 3836, 3837, 3838, 3839, 3840, 3841, 3842, 3843, 3844, 3845, 3846, 3847, 3848, 3849, 3850, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3863, 3864, 3865, 3866, 3867, 3868, 3869, 3871, 3872, 3874, 3875, 3876, 3877, 3878, 3880, 3881, 3882, 3884, 3885, 3886, 3887, 3891, 3892, 3893, 3895, 3896, 3898, 3899, 3900, 3901, 3902, 3903, 3905, 3906, 3907, 3908, 3910, 3911, 3912, 3913, 3914, 3915, 3916, 3917, 3918, 3919, 3920, 3921, 3922, 3923, 3924, 3925, 3926, 3927, 3928]

            #self.extended_mask = np.load('assets/mask.npy').tolist()
            self.extended_mask = np.load('assets/extended_forehead_mask.npy').tolist()
            self.face_mask = flame_masks['face']

            if config.render.extended:
                self.final_mask = self.extended_mask
            else:
                self.final_mask = self.face_mask.tolist()

            if config.render.different_color_extended:
                colors[:, self.extended_mask, :] = torch.tensor([255, 0, 0]).float()/255
                colors[:, self.face_mask, :] = torch.tensor([180, 180, 180]).float()/255
                


            if config.render.eyes:

                self.eyes = np.concatenate((flame_masks['left_eyeball'], flame_masks['right_eyeball']))
                # change colors of eyes
                # colors[:, self.eyes, :] = torch.tensor([255, 255, 255]).float()/255
                self.final_mask = np.concatenate((self.final_mask, self.eyes)).tolist()

            #else:
            #    self.final_mask = self.final_mask

                # pupils to black
                # self.pupils = [3931, 3932, 3935, 3936, 3939, 3940, 3943, 3944, 3947, 3948, 3951, 3952, 3955, 3956, 3959, 3960, 3963, 3964, 3967, 3968, 3971, 3972, 3975, 3976, 3979, 3980, 3983, 3984, 3987, 3988, 3991, 3992, 3995, 3996, 3999, 4000, 4003, 4004, 4007, 4008, 4011, 4012, 4015, 4016, 4019, 4020, 4023, 4024, 4027, 4028, 4031, 4032, 4035, 4036, 4039, 4040, 4043, 4044, 4047, 4048, 4051, 4052, 4053, 4056, 4057, 4477, 4478, 4481, 4482, 4485, 4486, 4489, 4490, 4493, 4494, 4497, 4498, 4501, 4502, 4505, 4506, 4509, 4510, 4513, 4514, 4517, 4518, 4521, 4522, 4525, 4526, 4529, 4530, 4533, 4534, 4537, 4538, 4541, 4542, 4545, 4546, 4549, 4550, 4553, 4554, 4557, 4558, 4561, 4562, 4565, 4566, 4569, 4570, 4573, 4574, 4577, 4578, 4581, 4582, 4585, 4586, 4589, 4590, 4593, 4594, 4597, 4598, 4599, 4602, 4603]
                # colors[:, self.pupils, :] = torch.tensor([10, 10, 10]).float()/255

                


            # keep only faces that include vertices in face_mask
            faces = keep_vertices_and_update_faces(faces[0], self.final_mask).unsqueeze(0)

            colors = colors[:, self.final_mask, :]





        use_colored_visualization = False
        if use_colored_visualization:
            import pickle
            flame_masks = pickle.load(
                open('assets/FLAME_masks/FLAME_masks.pkl', 'rb'),
                encoding='latin1')
            # self.face_mask = torch.from_numpy(flame_masks['face']).cuda()#,flame_masks['left_eyeball'], flame_masks['right_eyeball']))
            colors2[:, flame_masks['lips'], :] = torch.tensor([0, 255, 0]).float()/255
            colors2[:, flame_masks['nose'], :] = torch.tensor([0, 0, 255]).float()/255
            colors2[:, flame_masks['eye_region'], :] = torch.tensor([255, 255, 0]).float()/255
            colors2[:, flame_masks['forehead'], :] = torch.tensor([255, 0, 255]).float()/255

            colors = colors2[:, self.face_mask, :]

        use_nmfc = False
        if use_nmfc:
            self.nmfc = verts.view(1,-1,3)
            self.nmfc = (self.nmfc - self.nmfc.min(dim=1, keepdim=True)[0])/(self.nmfc.max(dim=1, keepdim=True)[0] - self.nmfc.min(dim=1, keepdim=True)[0])

            colors = self.nmfc[:, self.face_mask, :]



        self.register_buffer('faces', faces)

        face_colors = util.face_vertices(colors, faces)
        self.register_buffer('face_colors', face_colors)

        self.register_buffer('raw_uvcoords', uvcoords)

        # uv coords
        uvcoords = torch.cat([uvcoords, uvcoords[:,:,0:1]*0.+1.], -1) #[bz, ntv, 3]
        uvcoords = uvcoords*2 - 1; uvcoords[...,1] = -uvcoords[...,1]
        face_uvcoords = util.face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)

        ## SH factors for lighting
        pi = np.pi
        constant_factor = torch.tensor([1/np.sqrt(4*pi), ((2*pi)/3)*(np.sqrt(3/(4*pi))), ((2*pi)/3)*(np.sqrt(3/(4*pi))),\
                           ((2*pi)/3)*(np.sqrt(3/(4*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))),\
                           (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3/2)*(np.sqrt(5/(12*pi))), (pi/4)*(1/2)*(np.sqrt(5/(4*pi)))]).float()
        self.register_buffer('constant_factor', constant_factor)

    def forward(self, vertices, transformed_vertices, lights=None, colors=None):
        '''
        -- Texture Rendering
        vertices: [batch_size, V, 3], vertices in world space, for calculating normals, then shading
        transformed_vertices: [batch_size, V, 3], range:normalized to [-1,1], projected vertices in image space (that is aligned to the iamge pixel), for rasterization
        albedos: [batch_size, 3, h, w], uv map
        lights: 
            spherical homarnic: [N, 9(shcoeff), 3(rgb)]
            points/directional lighting: [N, n_lights, 6(xyzrgb)]
        light_type:
            point or directional
        '''
        batch_size = vertices.shape[0]

        light_positions = torch.tensor(
            [
            [-1,1,1],
            [1,1,1],
            [-1,-1,1],
            [1,-1,1],
            [0,0,1]
            ]
        )[None,:,:].expand(batch_size, -1, -1).float()
        light_intensities = torch.ones_like(light_positions).float()*1.7
        lights = torch.cat((light_positions, light_intensities), 2).to(vertices.device)
        
        if not self.config.render.full_head:
            transformed_vertices = transformed_vertices[:,self.final_mask,:]
            vertices = vertices[:,self.final_mask,:]

        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10
        # attributes
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1)) 
        face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        
        colors = self.face_colors.expand(batch_size, -1, -1, -1)

        attributes = torch.cat([colors,
                                face_normals], 
                                -1)
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)
        
        albedo_images = rendering[:, :3, :, :]

        # shading
        normal_images = rendering[:, 3:6, :, :]

        shading = self.add_directionlight(normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
        shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2).contiguous()        
        shaded_images = albedo_images*shading_images
        

        return shaded_images


    def add_directionlight(self, normals, lights):
        '''
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_direction = lights[:,:,:3]; light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_direction[:,:,None,:].expand(-1,-1,normals.shape[1],-1), dim=3)
        normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)




    def forward_with_albedo(self, vertices, transformed_vertices, albedos, lights=None, h=None, w=None, light_type='point', background=None):
        '''
        -- Texture Rendering
        vertices: [batch_size, V, 3], vertices in world space, for calculating normals, then shading
        transformed_vertices: [batch_size, V, 3], range:normalized to [-1,1], projected vertices in image space (that is aligned to the iamge pixel), for rasterization
        albedos: [batch_size, 3, h, w], uv map
        lights: 
            spherical homarnic: [N, 9(shcoeff), 3(rgb)]
            points/directional lighting: [N, n_lights, 6(xyzrgb)]
        light_type:
            point or directional
        '''
        batch_size = vertices.shape[0]
        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10
        # attributes
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1)); face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1)); transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        
        attributes = torch.cat([self.face_uvcoords.expand(batch_size, -1, -1, -1), 
                                transformed_face_normals.detach(), 
                                face_vertices.detach(), 
                                face_normals], 
                                -1)
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes, h, w)
        
        ####
        # vis mask
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        uvcoords_images = rendering[:, :3, :, :]; grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
        albedo_images = F.grid_sample(albedos, grid, align_corners=False)

        # visible mask for pixels with positive normal direction
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < -0.05).float()

        # shading
        normal_images = rendering[:, 9:12, :, :]
        if lights is not None:
            if lights.shape[1] == 9:
                shading_images = self.add_SHlight(normal_images, lights)
            else:
                if light_type=='point':
                    vertice_images = rendering[:, 6:9, :, :].detach()
                    shading = self.add_pointlight(vertice_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
                    shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2)
                else:
                    shading = self.add_directionlight(normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
                    shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2)
            images = albedo_images*shading_images
        else:
            images = albedo_images
            shading_images = images.detach()*0.

        if background is not None:
            images = images*alpha_images + background*(1.-alpha_images)
            albedo_images = albedo_images*alpha_images + background*(1.-alpha_images)
        else:
            images = images*alpha_images 
            albedo_images = albedo_images*alpha_images 

        # outputs = {
        #     'images': images,
        #     'albedo_images': albedo_images,
        #     'alpha_images': alpha_images,
        #     'pos_mask': pos_mask,
        #     'shading_images': shading_images,
        #     'grid': grid,
        #     'normals': normals,
        #     'normal_images': normal_images*alpha_images,
        #     'transformed_normals': transformed_normals,
        # }
        
        return images

    def add_SHlight(self, normal_images, sh_coeff):
        '''
            sh_coeff: [bz, 9, 3]
        '''
        N = normal_images
        sh = torch.stack([
                N[:,0]*0.+1., N[:,0], N[:,1], \
                N[:,2], N[:,0]*N[:,1], N[:,0]*N[:,2], 
                N[:,1]*N[:,2], N[:,0]**2 - N[:,1]**2, 3*(N[:,2]**2) - 1
                ], 
                1) # [bz, 9, h, w]
        sh = sh*self.constant_factor[None,:,None,None]
        shading = torch.sum(sh_coeff[:,:,:,None,None]*sh[:,:,None,:,:], 1) # [bz, 9, 3, h, w]  
        return shading

    def add_pointlight(self, vertices, normals, lights):
        '''
            vertices: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_positions = lights[:,:,:3]; light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_positions[:,:,None,:] - vertices[:,None,:,:], dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)
    

    def create_jet_colormap(self, errors, min_error=0, max_error=0.015):
        import matplotlib.pyplot as plt
        # Normalize errors to [0, 1]
        # min_val = torch.min(errors)
        # max_val = torch.max(errors)
        min_val = min_error
        max_val = max_error
        normalized_errors = (errors - min_val) / (max_val - min_val)

        # Apply jet colormap
        colormap = plt.get_cmap('jet')
        colored_errors = colormap(normalized_errors.cpu().numpy())  # Convert to numpy array for matplotlib

        # Convert RGBA to tensor and discard the alpha channel
        colored_errors_tensor = torch.from_numpy(colored_errors[..., :3]).float().cuda()

        black_mask = np.concatenate((self.flame_masks['scalp'], self.flame_masks['neck'], self.flame_masks['boundary'],
                                     self.flame_masks['left_ear'], self.flame_masks['left_eyeball'],
                                     self.flame_masks['right_ear'], self.flame_masks['right_eyeball'],
                                     ),axis=0)

        # Set the scalp region to black
        colored_errors_tensor[:, black_mask, :] = torch.tensor([0, 0, 0]).float().cuda()

        return colored_errors_tensor

    def render_errors(self, transformed_vertices, errors, pos_mask = None):
        '''
        -- rendering colors: could be rgb color/ normals, etc
            colors: [bz, num of vertices, 3]
        '''
        batch_size = errors.shape[0]

        # create colormap from errors
        colors = self.create_jet_colormap(errors)

        # Attributes
        attributes = util.face_vertices(colors, self.faces.expand(batch_size, -1, -1))
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        ####
        alpha_images = rendering[:, [-1], :, :].detach()
        images = rendering[:, :3, :, :]* alpha_images
        if pos_mask is not None:
            images = images*pos_mask
        return images