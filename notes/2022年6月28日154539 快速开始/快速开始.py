"""
@Auth: itmorn
@Date: 2022/6/28-15:46
@Email: 12567148@qq.com
"""

from paddleocr import PaddleOCR, draw_ocr

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
img_path = '../../doc/imgs/11.jpg'
result = ocr.ocr(img_path, cls=True)
for line in result:
    print(line)

# 显示结果
from PIL import Image

image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='../../doc/fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')

"""
download https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar to /root/.paddleocr/whl/det/ch/ch_PP-OCRv3_det_infer/ch_PP-OCRv3_det_infer.tar
100%|██████████████████████████████████████| 3.83M/3.83M [00:04<00:00, 795kiB/s]
download https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar to /root/.paddleocr/whl/rec/ch/ch_PP-OCRv3_rec_infer/ch_PP-OCRv3_rec_infer.tar
100%|█████████████████████████████████████| 11.9M/11.9M [00:08<00:00, 1.37MiB/s]
[2022/06/28 15:50:36] ppocr DEBUG: Namespace(help='==SUPPRESS==', use_gpu=True, use_xpu=False, ir_optim=True, use_tensorrt=False, min_subgraph_size=15, precision='fp32', gpu_mem=500, image_dir=None, det_algorithm='DB', det_model_dir='/root/.paddleocr/whl/det/ch/ch_PP-OCRv3_det_infer', det_limit_side_len=960, det_limit_type='max', det_db_thresh=0.3, det_db_box_thresh=0.6, det_db_unclip_ratio=1.5, max_batch_size=10, use_dilation=False, det_db_score_mode='fast', vis_seg_map=False, det_east_score_thresh=0.8, det_east_cover_thresh=0.1, det_east_nms_thresh=0.2, det_sast_score_thresh=0.5, det_sast_nms_thresh=0.2, det_sast_polygon=False, det_pse_thresh=0, det_pse_box_thresh=0.85, det_pse_min_area=16, det_pse_box_type='quad', det_pse_scale=1, scales=[8, 16, 32], alpha=1.0, beta=1.0, fourier_degree=5, det_fce_box_type='poly', rec_algorithm='SVTR_LCNet', rec_model_dir='/root/.paddleocr/whl/rec/ch/ch_PP-OCRv3_rec_infer', rec_image_shape='3, 48, 320', rec_batch_num=6, max_text_length=25, rec_char_dict_path='/data01/zhaoyichen/work_github/PaddleOCR/ppocr/utils/ppocr_keys_v1.txt', use_space_char=True, vis_font_path='./doc/fonts/simfang.ttf', drop_score=0.5, e2e_algorithm='PGNet', e2e_model_dir=None, e2e_limit_side_len=768, e2e_limit_type='max', e2e_pgnet_score_thresh=0.5, e2e_char_dict_path='./ppocr/utils/ic15_dict.txt', e2e_pgnet_valid_set='totaltext', e2e_pgnet_mode='fast', use_angle_cls=True, cls_model_dir='/root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer', cls_image_shape='3, 48, 192', label_list=['0', '180'], cls_batch_num=6, cls_thresh=0.9, enable_mkldnn=False, cpu_threads=10, use_pdserving=False, warmup=False, draw_img_save_dir='./inference_results', save_crop_res=False, crop_res_save_dir='./output', use_mp=False, total_process_num=1, process_id=0, benchmark=False, save_log_path='./log_output/', show_log=True, use_onnx=False, output='./output', table_max_len=488, table_model_dir=None, table_char_dict_path=None, layout_path_model='lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config', layout_label_map=None, mode='structure', layout=True, table=True, ocr=True, lang='ch', det=True, rec=True, type='ocr', ocr_version='PP-OCRv3', structure_version='PP-STRUCTURE')
[2022/06/28 15:50:42] ppocr DEBUG: dt_boxes num : 16, elapse : 3.411618709564209
[2022/06/28 15:50:42] ppocr DEBUG: cls num  : 16, elapse : 0.023383140563964844
[2022/06/28 15:50:42] ppocr DEBUG: rec_res num  : 16, elapse : 0.08866143226623535
[[[28.0, 37.0], [302.0, 39.0], [302.0, 72.0], [27.0, 70.0]], ('纯臻营养护发素', 0.9658740162849426)]
[[[26.0, 81.0], [172.0, 83.0], [172.0, 104.0], [25.0, 101.0]], ('产品信息/参数', 0.9113153219223022)]
[[[28.0, 115.0], [330.0, 115.0], [330.0, 132.0], [28.0, 132.0]], ('（45元/每公斤，100公斤起订）', 0.884330153465271)]
[[[27.0, 145.0], [282.0, 145.0], [282.0, 164.0], [27.0, 164.0]], ('每瓶22元，1000瓶起订）', 0.9211574792861938)]
[[[26.0, 179.0], [299.0, 179.0], [299.0, 195.0], [26.0, 195.0]], ('【品牌】：代加工方式/OEMODM', 0.9661449193954468)]
[[[26.0, 210.0], [233.0, 210.0], [233.0, 227.0], [26.0, 227.0]], ('【品名】：纯臻营养护发素', 0.8831901550292969)]
[[[26.0, 241.0], [241.0, 241.0], [241.0, 258.0], [26.0, 258.0]], ('【产品编号】：YM-X-3011', 0.8718007206916809)]
[[[413.0, 236.0], [430.0, 236.0], [430.0, 303.0], [413.0, 303.0]], ('ODMOEM', 0.9539499878883362)]
[[[23.0, 271.0], [180.0, 269.0], [180.0, 289.0], [24.0, 290.0]], ('【净含量】：220ml', 0.9348610043525696)]
[[[26.0, 304.0], [252.0, 304.0], [252.0, 320.0], [26.0, 320.0]], ('适用人群）：适合所有肤质', 0.8866128325462341)]
[[[26.0, 335.0], [343.0, 335.0], [343.0, 352.0], [26.0, 352.0]], ('【主要成分】：鲸蜡硬脂醇、燕麦β-葡聚', 0.9245769381523132)]
[[[27.0, 366.0], [281.0, 366.0], [281.0, 383.0], [27.0, 383.0]], ('糖、椰油酰胺丙基甜菜碱、泛醒', 0.9368143677711487)]
[[[369.0, 370.0], [477.0, 370.0], [477.0, 387.0], [369.0, 387.0]], ('（成品包材）', 0.8927481174468994)]
[[[26.0, 397.0], [361.0, 397.0], [361.0, 414.0], [26.0, 414.0]], ('【主要功能】：可紧致头发磷层，从而达到', 0.8677790760993958)]
[[[28.0, 429.0], [372.0, 429.0], [372.0, 445.0], [28.0, 445.0]], ('即时持久改善头发光泽的效果，给于燥的头', 0.8803289532661438)]
[[[27.0, 459.0], [136.0, 459.0], [136.0, 479.0], [27.0, 479.0]], ('发足够的滋养', 0.9091529846191406)]

"""