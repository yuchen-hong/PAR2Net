from util.util_test import *
from network.net_par2net import Net_PAR2Net

patch_num = 25
img_size = 32 * int(3 * np.sqrt(patch_num))

path_data = './data'
path_model = './model/par2net.pth.tar'


def resize_cube(img):
    h, w, _ = img.shape
    if h >= img_size:
        return cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    else:
        return cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)


def resize_pano(img, h):
    if h >= img_size:
        return cv2.resize(img, (h * 2, h), interpolation=cv2.INTER_AREA)
    else:
        return cv2.resize(img, (h * 2, h), interpolation=cv2.INTER_CUBIC)


def gen_cube_from_pano(glass, m):
    h, w, _ = m.shape

    list_point = np.argwhere(glass != 0)
    glass_cx = (np.max(list_point[:, 1]) + np.min(list_point[:, 1])) // 2
    delta = glass_cx - w // 2

    m = np.roll(m, axis=1, shift=-delta)
    glass = np.roll(glass, axis=1, shift=-delta)
    rs = np.roll(m[:, ::-1, :], axis=1, shift=w // 2)

    list_point = np.argwhere(glass != 0)
    glass_cx, glass_cy = w // 2, h // 2

    side_len = max(np.abs(np.max(list_point[:, 1]) - w // 2), np.abs(np.min(list_point[:, 1]) - w // 2),
                   np.abs(np.max(list_point[:, 0]) - h // 2), np.abs(np.min(list_point[:, 0]) - h // 2))
    side_len = side_len * 2

    glass_cube = glass[glass_cy - side_len // 2:glass_cy + side_len // 2,
                 glass_cx - side_len // 2:glass_cx + side_len // 2, :]
    m_cube = m[glass_cy - side_len // 2:glass_cy + side_len // 2, glass_cx - side_len // 2:glass_cx + side_len // 2, :]

    side_len_rs = int(side_len * 1.2)
    side_len_rs = min(side_len_rs, h)
    rs_cube = rs[glass_cy - side_len_rs // 2:glass_cy + side_len_rs // 2,
              glass_cx - side_len_rs // 2: glass_cx + side_len_rs // 2, :]

    list_return = []
    glass_cube = resize_cube(glass_cube)
    m_cube = resize_cube(m_cube)
    rs_cube = resize_cube(rs_cube)

    list_return.extend([m, glass, glass_cube, m_cube, rs_cube])

    return list_return, side_len


def warp_back_to_pano(cube, pano_m, pano_glass, side_len):
    h, w, _ = pano_m.shape
    h_new = int((img_size / side_len) * h)

    pano_m = resize_pano(pano_m, h_new)
    pano_glass = resize_pano(pano_glass, h_new)

    background = np.zeros(pano_m.shape, dtype=np.uint8)

    cx, cy = h_new, h_new // 2
    background[cy - img_size // 2:cy + img_size // 2, cx - img_size // 2:cx + img_size // 2, :] = cube.copy()
    pano_cube = np.where(pano_glass, background, pano_m)

    return pano_cube


def test():
    path_save = './result'
    path_save_r = osp.join(path_save, 'pred_r')
    path_save_t = osp.join(path_save, 'pred_t')

    make_dir(path_save_r)
    make_dir(path_save_t)

    model = Net_PAR2Net(n_feats=256, n_resblocks=13, patch_num=patch_num, reduction=8).cuda()
    model = torch.nn.DataParallel(model)

    list_data = os.listdir(path_data)
    checkpoint = torch.load(path_model)
    model.load_state_dict(checkpoint['state_dict'])

    with torch.no_grad():
        for index in list_data:
            if not index.isdigit():
                continue
            # =================== data loading ======================
            img_m = cv2.imread(osp.join(path_data, index, 'color.jpg'))
            img_glass = cv2.imread(osp.join(path_data, index, 'mask.jpg'))

            [ori_m, ori_glass, img_glass, img_m, img_rs], side_len = gen_cube_from_pano(img_glass, img_m)

            # =================== input ======================
            input_m = np2tensor(img_m, cuda=True)
            input_rs = np2tensor(img_rs, cuda=True)
            input_glass = np2tensor_wo_norm(img_glass[:, :, :1], cuda=True)

            # =================== output ======================
            pred_r, pred_t, _ = model(input_m, input_rs, input_glass)
            pred_r = tensor2np(pred_r)
            pred_t = tensor2np(pred_t)

            # ==================== calc ssim =====================
            img_m = img_m * img_glass

            # ==================== save =====================
            pred_r = warp_back_to_pano(pred_r, ori_m, ori_glass, side_len)
            pred_t = warp_back_to_pano(pred_t, ori_m, ori_glass, side_len)

            cv2.imwrite(osp.join(path_save_r, index + "_pred_r.jpg"), pred_r)
            cv2.imwrite(osp.join(path_save_t, index + "_pred_t.jpg"), pred_t)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    test()