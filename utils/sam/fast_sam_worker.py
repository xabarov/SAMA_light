from PySide2 import QtCore
from utils.sam.fast_sam_annotator import YoloSAM


class FastSAMWorker(QtCore.QThread):

    def __init__(self, sam_model_name='FastSAM-x.pt'):
        super(FastSAMWorker, self).__init__()
        self.sam = YoloSAM(sam_model_name)
        self.points = []
        self.pointlabel = []
        self.text_prompt = ""
        self.bbox = []

        self.source = None
        self.device = 'cpu'
        self.retina_mask = True
        self.imgsz = 1024
        self.conf = 0.4
        self.iou = 0.9

        self.shapes = []

    def set_image(self, source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9):
        self.source = source
        self.device = device
        self.retina_mask = retina_masks
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def everything(self):
        return self.sam.everything()

    def box(self, bbox):
        # Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
        return self.sam.box(bbox=bbox)

    def text(self, text_prompt):
        # Text prompt
        return self.sam.text(text_prompt)

    def point(self, points, pointlabel):
        # Point prompt
        # points default [[0,0]] [[x1,y1],[x2,y2]]
        # point_label default [0] [1,0] 0:background, 1:foreground
        return self.sam.point(points=points, pointlabel=pointlabel)

    def run(self):
        """
        Return shapes
        [{'id': 1, 'cls_num': 1, 'points': [[53.0, 561.0], [53.0, 562.0], [52.0, 562.0], [52.0, 564.0], [51.0, 564.0], [51.0, 565.0], [52.0, 565.0], [52.0, 571.0], [53.0, 571.0], [53.0, 572.0], [60.0, 572.0], [60.0, 571.0], [61.0, 571.0], [61.0, 569.0], [60.0, 569.0], [60.0, 567.0], [59.0, 567.0], [59.0, 566.0], [58.0, 566.0], [58.0, 565.0], [57.0, 565.0], [57.0, 564.0], [56.0, 564.0], [56.0, 563.0], [55.0, 563.0], [55.0, 562.0], [54.0, 562.0], [54.0, 561.0], [53.0, 561.0]]}, {'id': 2, 'cls_num': 2, 'points': [[599.0, 0.0], [599.0, 3.0], [595.0, 3.0], [595.0, 4.0], [504.0, 4.0], [504.0, 5.0], [502.0, 5.0], [502.0, 4.0], [493.0, 4.0], [493.0, 5.0], [449.0, 5.0], [449.0, 4.0], [403.0, 4.0], [403.0, 5.0], [400.0, 5.0], [400.0, 4.0], [379.0, 4.0], [379.0, 5.0], [376.0, 5.0], [376.0, 4.0], [369.0, 4.0], [369.0, 5.0], [183.0, 5.0], [183.0, 4.0], [168.0, 4.0], [168.0, 5.0], [154.0, 5.0], [154.0, 4.0], [127.0, 4.0], [127.0, 3.0], [74.0, 3.0], [74.0, 4.0], [66.0, 4.0], [66.0, 3.0], [64.0, 3.0], [64.0, 4.0], [39.0, 4.0], [39.0, 5.0], [7.0, 5.0], [7.0, 6.0], [6.0, 6.0], [6.0, 7.0], [4.0, 7.0], [4.0, 8.0], [3.0, 8.0], [3.0, 10.0], [0.0, 10.0], [0.0, 575.0], [3.0, 575.0], [3.0, 576.0], [4.0, 576.0], [4.0, 577.0], [5.0, 577.0], [5.0, 578.0], [18.0, 578.0], [18.0, 577.0], [20.0, 577.0], [20.0, 576.0], [21.0, 576.0], [21.0, 575.0], [22.0, 575.0], [22.0, 574.0], [23.0, 574.0], [23.0, 573.0], [24.0, 573.0], [24.0, 572.0], [25.0, 572.0], [25.0, 571.0], [26.0, 571.0], [26.0, 570.0], [27.0, 570.0], [27.0, 568.0], [28.0, 568.0], [28.0, 564.0], [29.0, 564.0], [29.0, 561.0], [30.0, 561.0], [30.0, 559.0], [31.0, 559.0], [31.0, 557.0], [32.0, 557.0], [32.0, 556.0], [33.0, 556.0], [33.0, 554.0], [34.0, 554.0], [34.0, 549.0], [35.0, 549.0], [35.0, 547.0], [36.0, 547.0], [36.0, 545.0], [38.0, 545.0], [38.0, 544.0], [39.0, 544.0], [39.0, 543.0], [40.0, 543.0], [40.0, 542.0], [41.0, 542.0], [41.0, 540.0], [44.0, 540.0], [44.0, 541.0], [45.0, 541.0], [45.0, 543.0], [46.0, 543.0], [46.0, 545.0], [47.0, 545.0], [47.0, 547.0], [48.0, 547.0], [48.0, 548.0], [49.0, 548.0], [49.0, 547.0], [51.0, 547.0], [51.0, 546.0], [54.0, 546.0], [54.0, 545.0], [55.0, 545.0], [55.0, 544.0], [56.0, 544.0], [56.0, 543.0], [57.0, 543.0], [57.0, 542.0], [58.0, 542.0], [58.0, 541.0], [59.0, 541.0], [59.0, 540.0], [61.0, 540.0], [61.0, 539.0], [63.0, 539.0], [63.0, 538.0], [64.0, 538.0], [64.0, 537.0], [66.0, 537.0], [66.0, 536.0], [67.0, 536.0], [67.0, 534.0], [68.0, 534.0], [68.0, 532.0], [69.0, 532.0], [69.0, 528.0], [70.0, 528.0], [70.0, 523.0], [71.0, 523.0], [71.0, 518.0], [72.0, 518.0], [72.0, 512.0], [73.0, 512.0], [73.0, 508.0], [74.0, 508.0], [74.0, 501.0], [75.0, 501.0], [75.0, 500.0], [76.0, 500.0], [76.0, 499.0], [78.0, 499.0], [78.0, 498.0], [79.0, 498.0], [79.0, 496.0], [80.0, 496.0], [80.0, 495.0], [81.0, 495.0], [81.0, 494.0], [83.0, 494.0], [83.0, 493.0], [84.0, 493.0], [84.0, 492.0], [85.0, 492.0], [85.0, 491.0], [86.0, 491.0], [86.0, 489.0], [87.0, 489.0], [87.0, 488.0], [89.0, 488.0], [89.0, 487.0], [90.0, 487.0], [90.0, 486.0], [91.0, 486.0], [91.0, 484.0], [92.0, 484.0], [92.0, 483.0], [94.0, 483.0], [94.0, 482.0], [100.0, 482.0], [100.0, 481.0], [104.0, 481.0], [104.0, 480.0], [106.0, 480.0], [106.0, 479.0], [111.0, 479.0], [111.0, 478.0], [113.0, 478.0], [113.0, 477.0], [116.0, 477.0], [116.0, 476.0], [118.0, 476.0], [118.0, 475.0], [119.0, 475.0], [119.0, 474.0], [120.0, 474.0], [120.0, 473.0], [121.0, 473.0], [121.0, 472.0], [122.0, 472.0], [122.0, 471.0], [124.0, 471.0], [124.0, 470.0], [125.0, 470.0], [125.0, 468.0], [126.0, 468.0], [126.0, 466.0], [128.0, 466.0], [128.0, 465.0], [130.0, 465.0], [130.0, 464.0], [132.0, 464.0], [132.0, 463.0], [134.0, 463.0], [134.0, 464.0], [135.0, 464.0], [135.0, 465.0], [136.0, 465.0], [136.0, 466.0], [137.0, 466.0], [137.0, 468.0], [138.0, 468.0], [138.0, 469.0], [139.0, 469.0], [139.0, 470.0], [140.0, 470.0], [140.0, 471.0], [141.0, 471.0], [141.0, 472.0], [142.0, 472.0], [142.0, 474.0], [143.0, 474.0], [143.0, 476.0], [144.0, 476.0], [144.0, 477.0], [146.0, 477.0], [146.0, 478.0], [148.0, 478.0], [148.0, 479.0], [149.0, 479.0], [149.0, 480.0], [150.0, 480.0], [150.0, 481.0], [151.0, 481.0], [151.0, 480.0], [152.0, 480.0], [152.0, 477.0], [153.0, 477.0], [153.0, 474.0], [154.0, 474.0], [154.0, 471.0], [155.0, 471.0], [155.0, 470.0], [156.0, 470.0], [156.0, 469.0], [157.0, 469.0], [157.0, 467.0], [158.0, 467.0], [158.0, 465.0], [159.0, 465.0], [159.0, 463.0], [160.0, 463.0], [160.0, 461.0], [161.0, 461.0], [161.0, 460.0], [163.0, 460.0], [163.0, 459.0], [164.0, 459.0], [164.0, 458.0], [165.0, 458.0], [165.0, 456.0], [166.0, 456.0], [166.0, 455.0], [167.0, 455.0], [167.0, 454.0], [169.0, 454.0], [169.0, 453.0], [170.0, 453.0], [170.0, 452.0], [171.0, 452.0], [171.0, 450.0], [172.0, 450.0], [172.0, 449.0], [174.0, 449.0], [174.0, 448.0], [177.0, 448.0], [177.0, 447.0], [179.0, 447.0], [179.0, 446.0], [180.0, 446.0], [180.0, 445.0], [181.0, 445.0], [181.0, 444.0], [183.0, 444.0], [183.0, 443.0], [185.0, 443.0], [185.0, 442.0], [191.0, 442.0], [191.0, 441.0], [192.0, 441.0], [192.0, 440.0], [193.0, 440.0], [193.0, 438.0], [195.0, 438.0], [195.0, 437.0], [196.0, 437.0], [196.0, 436.0], [198.0, 436.0], [198.0, 435.0], [199.0, 435.0], [199.0, 432.0], [200.0, 432.0], [200.0, 429.0], [201.0, 429.0], [201.0, 427.0], [202.0, 427.0], [202.0, 425.0], [203.0, 425.0], [203.0, 420.0], [204.0, 420.0], [204.0, 410.0], [205.0, 410.0], [205.0, 402.0], [206.0, 402.0], [206.0, 399.0], [207.0, 399.0], [207.0, 397.0], [208.0, 397.0], [208.0, 396.0], [209.0, 396.0], [209.0, 392.0], [210.0, 392.0], [210.0, 386.0], [211.0, 386.0], [211.0, 382.0], [212.0, 382.0], [212.0, 381.0], [213.0, 381.0], [213.0, 380.0], [215.0, 380.0], [215.0, 379.0], [217.0, 379.0], [217.0, 378.0], [218.0, 378.0], [218.0, 377.0], [219.0, 377.0], [219.0, 378.0], [220.0, 378.0], [220.0, 380.0], [221.0, 380.0], [221.0, 383.0], [222.0, 383.0], [222.0, 390.0], [223.0, 390.0], [223.0, 394.0], [224.0, 394.0], [224.0, 397.0], [225.0, 397.0], [225.0, 398.0], [226.0, 398.0], [226.0, 400.0], [227.0, 400.0], [227.0, 402.0], [228.0, 402.0], [228.0, 403.0], [229.0, 403.0], [229.0, 404.0], [231.0, 404.0], [231.0, 402.0], [232.0, 402.0], [232.0, 400.0], [233.0, 400.0], [233.0, 394.0], [234.0, 394.0], [234.0, 392.0], [235.0, 392.0], [235.0, 390.0], [236.0, 390.0], [236.0, 389.0], [237.0, 389.0], [237.0, 387.0], [238.0, 387.0], [238.0, 384.0], [239.0, 384.0], [239.0, 379.0], [240.0, 379.0], [240.0, 376.0], [241.0, 376.0], [241.0, 375.0], [242.0, 375.0], [242.0, 374.0], [243.0, 374.0], [243.0, 373.0], [244.0, 373.0], [244.0, 371.0], [245.0, 371.0], [245.0, 369.0], [246.0, 369.0], [246.0, 368.0], [247.0, 368.0], [247.0, 367.0], [248.0, 367.0], [248.0, 368.0], [250.0, 368.0], [250.0, 369.0], [251.0, 369.0], [251.0, 370.0], [256.0, 370.0], [256.0, 369.0], [258.0, 369.0], [258.0, 368.0], [259.0, 368.0], [259.0, 366.0], [260.0, 366.0], [260.0, 363.0], [261.0, 363.0], [261.0, 358.0], [262.0, 358.0], [262.0, 347.0], [261.0, 347.0], [261.0, 346.0], [260.0, 346.0], [260.0, 345.0], [259.0, 345.0], [259.0, 344.0], [258.0, 344.0], [258.0, 343.0], [257.0, 343.0], [257.0, 342.0], [255.0, 342.0], [255.0, 341.0], [252.0, 341.0], [252.0, 340.0], [240.0, 340.0], [240.0, 339.0], [239.0, 339.0], [239.0, 338.0], [238.0, 338.0], [238.0, 336.0], [239.0, 336.0], [239.0, 333.0], [240.0, 333.0], [240.0, 331.0], [241.0, 331.0], [241.0, 330.0], [244.0, 330.0], [244.0, 329.0], [246.0, 329.0], [246.0, 328.0], [248.0, 328.0], [248.0, 327.0], [249.0, 327.0], [249.0, 326.0], [250.0, 326.0], [250.0, 325.0], [251.0, 325.0], [251.0, 324.0], [254.0, 324.0], [254.0, 323.0], [256.0, 323.0], [256.0, 322.0], [257.0, 322.0], [257.0, 321.0], [258.0, 321.0], [258.0, 320.0], [260.0, 320.0], [260.0, 319.0], [261.0, 319.0], [261.0, 318.0], [263.0, 318.0], [263.0, 317.0], [265.0, 317.0], [265.0, 318.0], [266.0, 318.0], [266.0, 320.0], [267.0, 320.0], [267.0, 340.0], [266.0, 340.0], [266.0, 369.0], [265.0, 369.0], [265.0, 371.0], [264.0, 371.0], [264.0, 372.0], [265.0, 372.0], [265.0, 376.0], [266.0, 376.0], [266.0, 381.0], [267.0, 381.0], [267.0, 384.0], [268.0, 384.0], [268.0, 385.0], [269.0, 385.0], [269.0, 386.0], [271.0, 386.0], [271.0, 387.0], [272.0, 387.0], [272.0, 388.0], [273.0, 388.0], [273.0, 389.0], [274.0, 389.0], [274.0, 391.0], [276.0, 391.0], [276.0, 392.0], [277.0, 392.0], [277.0, 394.0], [278.0, 394.0], [278.0, 399.0], [279.0, 399.0], [279.0, 405.0], [280.0, 405.0], [280.0, 407.0], [281.0, 407.0], [281.0, 409.0], [282.0, 409.0], [282.0, 410.0], [283.0, 410.0], [283.0, 411.0], [284.0, 411.0], [284.0, 412.0], [285.0, 412.0], [285.0, 413.0], [286.0, 413.0], [286.0, 414.0], [288.0, 414.0], [288.0, 413.0], [289.0, 413.0], [289.0, 412.0], [290.0, 412.0], [290.0, 410.0], [291.0, 410.0], [291.0, 409.0], [292.0, 409.0], [292.0, 408.0], [293.0, 408.0], [293.0, 406.0], [294.0, 406.0], [294.0, 403.0], [295.0, 403.0], [295.0, 399.0], [296.0, 399.0], [296.0, 395.0], [297.0, 395.0], [297.0, 393.0], [298.0, 393.0], [298.0, 391.0], [299.0, 391.0], [299.0, 390.0], [300.0, 390.0], [300.0, 388.0], [301.0, 388.0], [301.0, 385.0], [302.0, 385.0], [302.0, 383.0], [303.0, 383.0], [303.0, 382.0], [305.0, 382.0], [305.0, 383.0], [306.0, 383.0], [306.0, 385.0], [307.0, 385.0], [307.0, 388.0], [308.0, 388.0], [308.0, 391.0], [309.0, 391.0], [309.0, 393.0], [310.0, 393.0], [310.0, 395.0], [311.0, 395.0], [311.0, 399.0], [312.0, 399.0], [312.0, 411.0], [313.0, 411.0], [313.0, 424.0], [314.0, 424.0], [314.0, 425.0], [315.0, 425.0], [315.0, 426.0], [316.0, 426.0], [316.0, 427.0], [317.0, 427.0], [317.0, 428.0], [318.0, 428.0], [318.0, 429.0], [319.0, 429.0], [319.0, 430.0], [322.0, 430.0], [322.0, 431.0], [327.0, 431.0], [327.0, 432.0], [331.0, 432.0], [331.0, 433.0], [333.0, 433.0], [333.0, 434.0], [334.0, 434.0], [334.0, 435.0], [336.0, 435.0], [336.0, 436.0], [340.0, 436.0], [340.0, 437.0], [351.0, 437.0], [351.0, 438.0], [355.0, 438.0], [355.0, 439.0], [356.0, 439.0], [356.0, 440.0], [358.0, 440.0], [358.0, 441.0], [360.0, 441.0], [360.0, 442.0], [370.0, 442.0], [370.0, 443.0], [378.0, 443.0], [378.0, 444.0], [381.0, 444.0], [381.0, 445.0], [387.0, 445.0], [387.0, 444.0], [390.0, 444.0], [390.0, 443.0], [391.0, 443.0], [391.0, 442.0], [392.0, 442.0], [392.0, 441.0], [393.0, 441.0], [393.0, 439.0], [394.0, 439.0], [394.0, 437.0], [395.0, 437.0], [395.0, 435.0], [396.0, 435.0], [396.0, 431.0], [397.0, 431.0], [397.0, 424.0], [398.0, 424.0], [398.0, 416.0], [399.0, 416.0], [399.0, 413.0], [400.0, 413.0], [400.0, 412.0], [401.0, 412.0], [401.0, 410.0], [402.0, 410.0], [402.0, 408.0], [403.0, 408.0], [403.0, 404.0], [404.0, 404.0], [404.0, 399.0], [405.0, 399.0], [405.0, 397.0], [406.0, 397.0], [406.0, 395.0], [407.0, 395.0], [407.0, 393.0], [408.0, 393.0], [408.0, 390.0], [409.0, 390.0], [409.0, 383.0], [410.0, 383.0], [410.0, 378.0], [411.0, 378.0], [411.0, 375.0], [412.0, 375.0], [412.0, 373.0], [413.0, 373.0], [413.0, 371.0], [414.0, 371.0], [414.0, 365.0], [415.0, 365.0], [415.0, 354.0], [416.0, 354.0], [416.0, 352.0], [417.0, 352.0], [417.0, 351.0], [419.0, 351.0], [419.0, 353.0], [420.0, 353.0], [420.0, 360.0], [421.0, 360.0], [421.0, 382.0], [422.0, 382.0], [422.0, 387.0], [423.0, 387.0], [423.0, 391.0], [424.0, 391.0], [424.0, 394.0], [425.0, 394.0], [425.0, 399.0], [426.0, 399.0], [426.0, 408.0], [427.0, 408.0], [427.0, 416.0], [428.0, 416.0], [428.0, 419.0], [429.0, 419.0], [429.0, 420.0], [430.0, 420.0], [430.0, 423.0], [431.0, 423.0], [431.0, 428.0], [432.0, 428.0], [432.0, 451.0], [433.0, 451.0], [433.0, 453.0], [434.0, 453.0], [434.0, 454.0], [435.0, 454.0], [435.0, 455.0], [436.0, 455.0], [436.0, 456.0], [437.0, 456.0], [437.0, 457.0], [438.0, 457.0], [438.0, 458.0], [439.0, 458.0], [439.0, 459.0], [441.0, 459.0], [441.0, 460.0], [445.0, 460.0], [445.0, 461.0], [447.0, 461.0], [447.0, 462.0], [448.0, 462.0], [448.0, 463.0], [450.0, 463.0], [450.0, 464.0], [452.0, 464.0], [452.0, 465.0], [464.0, 465.0], [464.0, 466.0], [468.0, 466.0], [468.0, 467.0], [470.0, 467.0], [470.0, 468.0], [471.0, 468.0], [471.0, 469.0], [473.0, 469.0], [473.0, 470.0], [477.0, 470.0], [477.0, 471.0], [486.0, 471.0], [486.0, 472.0], [488.0, 472.0], [488.0, 473.0], [490.0, 473.0], [490.0, 474.0], [491.0, 474.0], [491.0, 475.0], [495.0, 475.0], [495.0, 476.0], [516.0, 476.0], [516.0, 475.0], [517.0, 475.0], [517.0, 474.0], [518.0, 474.0], [518.0, 472.0], [519.0, 472.0], [519.0, 470.0], [520.0, 470.0], [520.0, 468.0], [521.0, 468.0], [521.0, 453.0], [522.0, 453.0], [522.0, 446.0], [523.0, 446.0], [523.0, 444.0], [522.0, 444.0], [522.0, 437.0], [523.0, 437.0], [523.0, 418.0], [524.0, 418.0], [524.0, 414.0], [525.0, 414.0], [525.0, 411.0], [526.0, 411.0], [526.0, 409.0], [527.0, 409.0], [527.0, 406.0], [528.0, 406.0], [528.0, 397.0], [529.0, 397.0], [529.0, 388.0], [530.0, 388.0], [530.0, 385.0], [531.0, 385.0], [531.0, 383.0], [537.0, 383.0], [537.0, 384.0], [538.0, 384.0], [538.0, 386.0], [539.0, 386.0], [539.0, 390.0], [540.0, 390.0], [540.0, 402.0], [541.0, 402.0], [541.0, 408.0], [542.0, 408.0], [542.0, 411.0], [543.0, 411.0], [543.0, 413.0], [544.0, 413.0], [544.0, 415.0], [545.0, 415.0], [545.0, 418.0], [546.0, 418.0], [546.0, 420.0], [547.0, 420.0], [547.0, 422.0], [548.0, 422.0], [548.0, 423.0], [549.0, 423.0], [549.0, 422.0], [551.0, 422.0], [551.0, 421.0], [553.0, 421.0], [553.0, 420.0], [563.0, 420.0], [563.0, 421.0], [566.0, 421.0], [566.0, 420.0], [570.0, 420.0], [570.0, 419.0], [572.0, 419.0], [572.0, 418.0], [573.0, 418.0], [573.0, 415.0], [574.0, 415.0], [574.0, 406.0], [575.0, 406.0], [575.0, 403.0], [576.0, 403.0], [576.0, 400.0], [577.0, 400.0], [577.0, 398.0], [578.0, 398.0], [578.0, 394.0], [579.0, 394.0], [579.0, 389.0], [580.0, 389.0], [580.0, 379.0], [581.0, 379.0], [581.0, 376.0], [582.0, 376.0], [582.0, 375.0], [583.0, 375.0], [583.0, 374.0], [584.0, 374.0], [584.0, 372.0], [585.0, 372.0], [585.0, 370.0], [586.0, 370.0], [586.0, 368.0], [587.0, 368.0], [587.0, 367.0], [588.0, 367.0], [588.0, 366.0], [589.0, 366.0], [589.0, 368.0], [590.0, 368.0], [590.0, 370.0], [591.0, 370.0], [591.0, 379.0], [592.0, 379.0], [592.0, 382.0], [593.0, 382.0], [593.0, 384.0], [594.0, 384.0], [594.0, 385.0], [595.0, 385.0], [595.0, 387.0], [596.0, 387.0], [596.0, 395.0], [597.0, 395.0], [597.0, 397.0], [598.0, 397.0], [598.0, 398.0], [599.0, 398.0], [599.0, 399.0], [600.0, 399.0], [600.0, 400.0], [601.0, 400.0], [601.0, 401.0], [602.0, 401.0], [602.0, 402.0], [603.0, 402.0], [603.0, 403.0], [604.0, 403.0], [604.0, 404.0], [605.0, 404.0], [605.0, 405.0], [606.0, 405.0], [606.0, 406.0], [607.0, 406.0], [607.0, 408.0], [608.0, 408.0], [608.0, 410.0], [609.0, 410.0], [609.0, 412.0], [610.0, 412.0], [610.0, 413.0], [611.0, 413.0], [611.0, 414.0], [613.0, 414.0], [613.0, 415.0], [616.0, 415.0], [616.0, 416.0], [623.0, 416.0], [623.0, 417.0], [624.0, 417.0], [624.0, 419.0], [625.0, 419.0], [625.0, 420.0], [626.0, 420.0], [626.0, 422.0], [627.0, 422.0], [627.0, 423.0], [628.0, 423.0], [628.0, 424.0], [629.0, 424.0], [629.0, 425.0], [630.0, 425.0], [630.0, 426.0], [631.0, 426.0], [631.0, 427.0], [632.0, 427.0], [632.0, 429.0], [633.0, 429.0], [633.0, 430.0], [637.0, 430.0], [637.0, 431.0], [640.0, 431.0], [640.0, 429.0], [641.0, 429.0], [641.0, 427.0], [642.0, 427.0], [642.0, 425.0], [643.0, 425.0], [643.0, 422.0], [644.0, 422.0], [644.0, 419.0], [645.0, 419.0], [645.0, 418.0], [646.0, 418.0], [646.0, 415.0], [647.0, 415.0], [647.0, 412.0], [648.0, 412.0], [648.0, 405.0], [649.0, 405.0], [649.0, 403.0], [650.0, 403.0], [650.0, 402.0], [651.0, 402.0], [651.0, 400.0], [652.0, 400.0], [652.0, 398.0], [653.0, 398.0], [653.0, 395.0], [654.0, 395.0], [654.0, 392.0], [655.0, 392.0], [655.0, 390.0], [656.0, 390.0], [656.0, 389.0], [657.0, 389.0], [657.0, 388.0], [659.0, 388.0], [659.0, 387.0], [660.0, 387.0], [660.0, 386.0], [661.0, 386.0], [661.0, 385.0], [662.0, 385.0], [662.0, 383.0], [663.0, 383.0], [663.0, 381.0], [664.0, 381.0], [664.0, 379.0], [665.0, 379.0], [665.0, 376.0], [666.0, 376.0], [666.0, 373.0], [667.0, 373.0], [667.0, 371.0], [668.0, 371.0], [668.0, 369.0], [669.0, 369.0], [669.0, 367.0], [670.0, 367.0], [670.0, 362.0], [671.0, 362.0], [671.0, 355.0], [672.0, 355.0], [672.0, 352.0], [673.0, 352.0], [673.0, 350.0], [675.0, 350.0], [675.0, 349.0], [676.0, 349.0], [676.0, 348.0], [678.0, 348.0], [678.0, 347.0], [680.0, 347.0], [680.0, 349.0], [681.0, 349.0], [681.0, 354.0], [682.0, 354.0], [682.0, 366.0], [683.0, 366.0], [683.0, 370.0], [684.0, 370.0], [684.0, 375.0], [685.0, 375.0], [685.0, 378.0], [686.0, 378.0], [686.0, 381.0], [687.0, 381.0], [687.0, 389.0], [688.0, 389.0], [688.0, 397.0], [689.0, 397.0], [689.0, 399.0], [690.0, 399.0], [690.0, 401.0], [694.0, 401.0], [694.0, 400.0], [697.0, 400.0], [697.0, 399.0], [698.0, 399.0], [698.0, 398.0], [699.0, 398.0], [699.0, 397.0], [700.0, 397.0], [700.0, 396.0], [701.0, 396.0], [701.0, 395.0], [702.0, 395.0], [702.0, 394.0], [703.0, 394.0], [703.0, 393.0], [704.0, 393.0], [704.0, 392.0], [706.0, 392.0], [706.0, 391.0], [707.0, 391.0], [707.0, 390.0], [709.0, 390.0], [709.0, 391.0], [714.0, 391.0], [714.0, 392.0], [718.0, 392.0], [718.0, 393.0], [720.0, 393.0], [720.0, 392.0], [723.0, 392.0], [723.0, 391.0], [725.0, 391.0], [725.0, 390.0], [726.0, 390.0], [726.0, 389.0], [727.0, 389.0], [727.0, 388.0], [728.0, 388.0], [728.0, 386.0], [729.0, 386.0], [729.0, 384.0], [730.0, 384.0], [730.0, 383.0], [731.0, 383.0], [731.0, 380.0], [732.0, 380.0], [732.0, 377.0], [733.0, 377.0], [733.0, 365.0], [734.0, 365.0], [734.0, 359.0], [735.0, 359.0], [735.0, 354.0], [736.0, 354.0], [736.0, 353.0], [737.0, 353.0], [737.0, 351.0], [738.0, 351.0], [738.0, 346.0], [739.0, 346.0], [739.0, 340.0], [740.0, 340.0], [740.0, 337.0], [741.0, 337.0], [741.0, 335.0], [743.0, 335.0], [743.0, 334.0], [744.0, 334.0], [744.0, 333.0], [745.0, 333.0], [745.0, 332.0], [746.0, 332.0], [746.0, 331.0], [748.0, 331.0], [748.0, 332.0], [749.0, 332.0], [749.0, 337.0], [750.0, 337.0], [750.0, 349.0], [751.0, 349.0], [751.0, 355.0], [752.0, 355.0], [752.0, 359.0], [753.0, 359.0], [753.0, 363.0], [754.0, 363.0], [754.0, 367.0], [755.0, 367.0], [755.0, 372.0], [756.0, 372.0], [756.0, 387.0], [757.0, 387.0], [757.0, 395.0], [758.0, 395.0], [758.0, 399.0], [759.0, 399.0], [759.0, 400.0], [760.0, 400.0], [760.0, 401.0], [761.0, 401.0], [761.0, 402.0], [764.0, 402.0], [764.0, 403.0], [765.0, 403.0], [765.0, 402.0], [766.0, 402.0], [766.0, 401.0], [767.0, 401.0], [767.0, 400.0], [768.0, 400.0], [768.0, 399.0], [769.0, 399.0], [769.0, 398.0], [770.0, 398.0], [770.0, 397.0], [771.0, 397.0], [771.0, 395.0], [772.0, 395.0], [772.0, 393.0], [773.0, 393.0], [773.0, 391.0], [774.0, 391.0], [774.0, 389.0], [775.0, 389.0], [775.0, 388.0], [776.0, 388.0], [776.0, 387.0], [777.0, 387.0], [777.0, 388.0], [782.0, 388.0], [782.0, 390.0], [783.0, 390.0], [783.0, 393.0], [784.0, 393.0], [784.0, 397.0], [785.0, 397.0], [785.0, 401.0], [786.0, 401.0], [786.0, 404.0], [787.0, 404.0], [787.0, 406.0], [788.0, 406.0], [788.0, 407.0], [789.0, 407.0], [789.0, 408.0], [790.0, 408.0], [790.0, 409.0], [792.0, 409.0], [792.0, 410.0], [793.0, 410.0], [793.0, 409.0], [794.0, 409.0], [794.0, 407.0], [795.0, 407.0], [795.0, 402.0], [796.0, 402.0], [796.0, 397.0], [797.0, 397.0], [797.0, 394.0], [798.0, 394.0], [798.0, 392.0], [799.0, 392.0], [799.0, 390.0], [800.0, 390.0], [800.0, 386.0], [801.0, 386.0], [801.0, 378.0], [802.0, 378.0], [802.0, 371.0], [803.0, 371.0], [803.0, 369.0], [804.0, 369.0], [804.0, 368.0], [805.0, 368.0], [805.0, 367.0], [806.0, 367.0], [806.0, 364.0], [807.0, 364.0], [807.0, 362.0], [808.0, 362.0], [808.0, 360.0], [809.0, 360.0], [809.0, 359.0], [811.0, 359.0], [811.0, 360.0], [812.0, 360.0], [812.0, 362.0], [813.0, 362.0], [813.0, 365.0], [814.0, 365.0], [814.0, 368.0], [815.0, 368.0], [815.0, 371.0], [816.0, 371.0], [816.0, 373.0], [817.0, 373.0], [817.0, 376.0], [818.0, 376.0], [818.0, 389.0], [819.0, 389.0], [819.0, 514.0], [820.0, 514.0], [820.0, 519.0], [821.0, 519.0], [821.0, 520.0], [823.0, 520.0], [823.0, 521.0], [826.0, 521.0], [826.0, 522.0], [827.0, 522.0], [827.0, 521.0], [829.0, 521.0], [829.0, 520.0], [830.0, 520.0], [830.0, 518.0], [831.0, 518.0], [831.0, 516.0], [832.0, 516.0], [832.0, 515.0], [833.0, 515.0], [833.0, 514.0], [834.0, 514.0], [834.0, 511.0], [835.0, 511.0], [835.0, 506.0], [836.0, 506.0], [836.0, 501.0], [837.0, 501.0], [837.0, 499.0], [838.0, 499.0], [838.0, 498.0], [842.0, 498.0], [842.0, 499.0], [844.0, 499.0], [844.0, 500.0], [845.0, 500.0], [845.0, 501.0], [846.0, 501.0], [846.0, 506.0], [847.0, 506.0], [847.0, 513.0], [848.0, 513.0], [848.0, 516.0], [849.0, 516.0], [849.0, 517.0], [850.0, 517.0], [850.0, 518.0], [852.0, 518.0], [852.0, 519.0], [853.0, 519.0], [853.0, 520.0], [855.0, 520.0], [855.0, 521.0], [856.0, 521.0], [856.0, 520.0], [859.0, 520.0], [859.0, 519.0], [861.0, 519.0], [861.0, 518.0], [862.0, 518.0], [862.0, 517.0], [863.0, 517.0], [863.0, 515.0], [864.0, 515.0], [864.0, 513.0], [865.0, 513.0], [865.0, 511.0], [866.0, 511.0], [866.0, 510.0], [867.0, 510.0], [867.0, 509.0], [869.0, 509.0], [869.0, 508.0], [871.0, 508.0], [871.0, 507.0], [873.0, 507.0], [873.0, 508.0], [874.0, 508.0], [874.0, 511.0], [875.0, 511.0], [875.0, 514.0], [876.0, 514.0], [876.0, 516.0], [877.0, 516.0], [877.0, 518.0], [878.0, 518.0], [878.0, 519.0], [879.0, 519.0], [879.0, 520.0], [880.0, 520.0], [880.0, 521.0], [881.0, 521.0], [881.0, 522.0], [882.0, 522.0], [882.0, 523.0], [883.0, 523.0], [883.0, 524.0], [887.0, 524.0], [887.0, 523.0], [890.0, 523.0], [890.0, 522.0], [891.0, 522.0], [891.0, 521.0], [892.0, 521.0], [892.0, 518.0], [893.0, 518.0], [893.0, 515.0], [894.0, 515.0], [894.0, 513.0], [897.0, 513.0], [897.0, 514.0], [898.0, 514.0], [898.0, 515.0], [900.0, 515.0], [900.0, 516.0], [901.0, 516.0], [901.0, 517.0], [902.0, 517.0], [902.0, 519.0], [903.0, 519.0], [903.0, 522.0], [904.0, 522.0], [904.0, 526.0], [905.0, 526.0], [905.0, 528.0], [906.0, 528.0], [906.0, 530.0], [907.0, 530.0], [907.0, 531.0], [908.0, 531.0], [908.0, 532.0], [909.0, 532.0], [909.0, 533.0], [910.0, 533.0], [910.0, 534.0], [912.0, 534.0], [912.0, 535.0], [913.0, 535.0], [913.0, 533.0], [914.0, 533.0], [914.0, 531.0], [915.0, 531.0], [915.0, 528.0], [916.0, 528.0], [916.0, 526.0], [917.0, 526.0], [917.0, 524.0], [918.0, 524.0], [918.0, 523.0], [919.0, 523.0], [919.0, 522.0], [920.0, 522.0], [920.0, 521.0], [921.0, 521.0], [921.0, 520.0], [922.0, 520.0], [922.0, 519.0], [923.0, 519.0], [923.0, 518.0], [924.0, 518.0], [924.0, 519.0], [925.0, 519.0], [925.0, 520.0], [926.0, 520.0], [926.0, 521.0], [927.0, 521.0], [927.0, 522.0], [928.0, 522.0], [928.0, 523.0], [929.0, 523.0], [929.0, 524.0], [930.0, 524.0], [930.0, 525.0], [931.0, 525.0], [931.0, 526.0], [932.0, 526.0], [932.0, 528.0], [933.0, 528.0], [933.0, 529.0], [934.0, 529.0], [934.0, 531.0], [937.0, 531.0], [937.0, 530.0], [941.0, 530.0], [941.0, 529.0], [944.0, 529.0], [944.0, 528.0], [947.0, 528.0], [947.0, 529.0], [948.0, 529.0], [948.0, 530.0], [949.0, 530.0], [949.0, 532.0], [950.0, 532.0], [950.0, 533.0], [951.0, 533.0], [951.0, 535.0], [952.0, 535.0], [952.0, 536.0], [953.0, 536.0], [953.0, 537.0], [954.0, 537.0], [954.0, 538.0], [955.0, 538.0], [955.0, 539.0], [956.0, 539.0], [956.0, 540.0], [957.0, 540.0], [957.0, 541.0], [959.0, 541.0], [959.0, 540.0], [960.0, 540.0], [960.0, 539.0], [961.0, 539.0], [961.0, 538.0], [962.0, 538.0], [962.0, 537.0], [963.0, 537.0], [963.0, 536.0], [964.0, 536.0], [964.0, 535.0], [966.0, 535.0], [966.0, 534.0], [967.0, 534.0], [967.0, 533.0], [970.0, 533.0], [970.0, 534.0], [972.0, 534.0], [972.0, 536.0], [973.0, 536.0], [973.0, 537.0], [974.0, 537.0], [974.0, 538.0], [975.0, 538.0], [975.0, 539.0], [977.0, 539.0], [977.0, 540.0], [978.0, 540.0], [978.0, 541.0], [979.0, 541.0], [979.0, 542.0], [980.0, 542.0], [980.0, 543.0], [981.0, 543.0], [981.0, 540.0], [982.0, 540.0], [982.0, 537.0], [983.0, 537.0], [983.0, 534.0], [984.0, 534.0], [984.0, 532.0], [985.0, 532.0], [985.0, 531.0], [986.0, 531.0], [986.0, 529.0], [987.0, 529.0], [987.0, 527.0], [988.0, 527.0], [988.0, 522.0], [989.0, 522.0], [989.0, 516.0], [990.0, 516.0], [990.0, 512.0], [991.0, 512.0], [991.0, 511.0], [992.0, 511.0], [992.0, 509.0], [993.0, 509.0], [993.0, 507.0], [994.0, 507.0], [994.0, 497.0], [995.0, 497.0], [995.0, 485.0], [996.0, 485.0], [996.0, 483.0], [997.0, 483.0], [997.0, 482.0], [998.0, 482.0], [998.0, 481.0], [999.0, 481.0], [999.0, 480.0], [1000.0, 480.0], [1000.0, 478.0], [1001.0, 478.0], [1001.0, 476.0], [1002.0, 476.0], [1002.0, 474.0], [1003.0, 474.0], [1003.0, 473.0], [1004.0, 473.0], [1004.0, 472.0], [1006.0, 472.0], [1006.0, 471.0], [1007.0, 471.0], [1007.0, 470.0], [1008.0, 470.0], [1008.0, 469.0], [1010.0, 469.0], [1010.0, 467.0], [1012.0, 467.0], [1012.0, 466.0], [1013.0, 466.0], [1013.0, 465.0], [1016.0, 465.0], [1016.0, 464.0], [1018.0, 464.0], [1018.0, 463.0], [1019.0, 463.0], [1019.0, 462.0], [1020.0, 462.0], [1020.0, 461.0], [1022.0, 461.0], [1022.0, 460.0], [1026.0, 460.0], [1026.0, 459.0], [1033.0, 459.0], [1033.0, 458.0], [1034.0, 458.0], [1034.0, 457.0], [1035.0, 457.0], [1035.0, 456.0], [1036.0, 456.0], [1036.0, 455.0], [1038.0, 455.0], [1038.0, 454.0], [1047.0, 454.0], [1047.0, 453.0], [1051.0, 453.0], [1051.0, 452.0], [1052.0, 452.0], [1052.0, 451.0], [1054.0, 451.0], [1054.0, 450.0], [1055.0, 450.0], [1055.0, 449.0], [1059.0, 449.0], [1059.0, 448.0], [1065.0, 448.0], [1065.0, 447.0], [1067.0, 447.0], [1067.0, 446.0], [1068.0, 446.0], [1068.0, 445.0], [1069.0, 445.0], [1069.0, 444.0], [1070.0, 444.0], [1070.0, 443.0], [1071.0, 443.0], [1071.0, 442.0], [1072.0, 442.0], [1072.0, 441.0], [1073.0, 441.0], [1073.0, 440.0], [1074.0, 440.0], [1074.0, 374.0], [1073.0, 374.0], [1073.0, 367.0], [1072.0, 367.0], [1072.0, 363.0], [1071.0, 363.0], [1071.0, 359.0], [1070.0, 359.0], [1070.0, 333.0], [1071.0, 333.0], [1071.0, 319.0], [1070.0, 319.0], [1070.0, 297.0], [1069.0, 297.0], [1069.0, 274.0], [1070.0, 274.0], [1070.0, 272.0], [1071.0, 272.0], [1071.0, 271.0], [1072.0, 271.0], [1072.0, 269.0], [1073.0, 269.0], [1073.0, 265.0], [1074.0, 265.0], [1074.0, 249.0], [1075.0, 249.0], [1075.0, 241.0], [1076.0, 241.0], [1076.0, 238.0], [1077.0, 238.0], [1077.0, 236.0], [1078.0, 236.0], [1078.0, 223.0], [1079.0, 223.0], [1079.0, 220.0], [1080.0, 220.0], [1080.0, 218.0], [1081.0, 218.0], [1081.0, 216.0], [1083.0, 216.0], [1083.0, 215.0], [1087.0, 215.0], [1087.0, 214.0], [1089.0, 214.0], [1089.0, 213.0], [1090.0, 213.0], [1090.0, 212.0], [1091.0, 212.0], [1091.0, 211.0], [1093.0, 211.0], [1093.0, 210.0], [1094.0, 210.0], [1094.0, 209.0], [1095.0, 209.0], [1095.0, 208.0], [1096.0, 208.0], [1096.0, 207.0], [1097.0, 207.0], [1097.0, 205.0], [1098.0, 205.0], [1098.0, 204.0], [1099.0, 204.0], [1099.0, 203.0], [1100.0, 203.0], [1100.0, 202.0], [1101.0, 202.0], [1101.0, 200.0], [1102.0, 200.0], [1102.0, 193.0], [1103.0, 193.0], [1103.0, 160.0], [1104.0, 160.0], [1104.0, 155.0], [1105.0, 155.0], [1105.0, 153.0], [1106.0, 153.0], [1106.0, 151.0], [1107.0, 151.0], [1107.0, 146.0], [1108.0, 146.0], [1108.0, 138.0], [1109.0, 138.0], [1109.0, 133.0], [1110.0, 133.0], [1110.0, 132.0], [1111.0, 132.0], [1111.0, 131.0], [1112.0, 131.0], [1112.0, 130.0], [1113.0, 130.0], [1113.0, 129.0], [1114.0, 129.0], [1114.0, 128.0], [1115.0, 128.0], [1115.0, 127.0], [1116.0, 127.0], [1116.0, 126.0], [1117.0, 126.0], [1117.0, 125.0], [1118.0, 125.0], [1118.0, 124.0], [1119.0, 124.0], [1119.0, 122.0], [1120.0, 122.0], [1120.0, 117.0], [1121.0, 117.0], [1121.0, 115.0], [1122.0, 115.0], [1122.0, 114.0], [1123.0, 114.0], [1123.0, 113.0], [1124.0, 113.0], [1124.0, 111.0], [1125.0, 111.0], [1125.0, 106.0], [1126.0, 106.0], [1126.0, 98.0], [1127.0, 98.0], [1127.0, 95.0], [1128.0, 95.0], [1128.0, 93.0], [1129.0, 93.0], [1129.0, 91.0], [1130.0, 91.0], [1130.0, 86.0], [1131.0, 86.0], [1131.0, 52.0], [1132.0, 52.0], [1132.0, 45.0], [1133.0, 45.0], [1133.0, 42.0], [1134.0, 42.0], [1134.0, 41.0], [1135.0, 41.0], [1135.0, 40.0], [1136.0, 40.0], [1136.0, 38.0], [1137.0, 38.0], [1137.0, 37.0], [1138.0, 37.0], [1138.0, 36.0], [1139.0, 36.0], [1139.0, 35.0], [1140.0, 35.0], [1140.0, 36.0], [1141.0, 36.0], [1141.0, 39.0], [1142.0, 39.0], [1142.0, 46.0], [1143.0, 46.0], [1143.0, 76.0], [1144.0, 76.0], [1144.0, 79.0], [1145.0, 79.0], [1145.0, 81.0], [1146.0, 81.0], [1146.0, 84.0], [1147.0, 84.0], [1147.0, 87.0], [1148.0, 87.0], [1148.0, 101.0], [1149.0, 101.0], [1149.0, 106.0], [1150.0, 106.0], [1150.0, 108.0], [1151.0, 108.0], [1151.0, 109.0], [1152.0, 109.0], [1152.0, 111.0], [1153.0, 111.0], [1153.0, 116.0], [1154.0, 116.0], [1154.0, 122.0], [1155.0, 122.0], [1155.0, 123.0], [1156.0, 123.0], [1156.0, 124.0], [1157.0, 124.0], [1157.0, 125.0], [1158.0, 125.0], [1158.0, 126.0], [1159.0, 126.0], [1159.0, 129.0], [1160.0, 129.0], [1160.0, 132.0], [1161.0, 132.0], [1161.0, 134.0], [1162.0, 134.0], [1162.0, 135.0], [1163.0, 135.0], [1163.0, 136.0], [1164.0, 136.0], [1164.0, 138.0], [1165.0, 138.0], [1165.0, 142.0], [1166.0, 142.0], [1166.0, 145.0], [1167.0, 145.0], [1167.0, 146.0], [1168.0, 146.0], [1168.0, 147.0], [1169.0, 147.0], [1169.0, 149.0], [1170.0, 149.0], [1170.0, 153.0], [1171.0, 153.0], [1171.0, 162.0], [1172.0, 162.0], [1172.0, 182.0], [1173.0, 182.0], [1173.0, 192.0], [1174.0, 192.0], [1174.0, 195.0], [1175.0, 195.0], [1175.0, 198.0], [1176.0, 198.0], [1176.0, 203.0], [1177.0, 203.0], [1177.0, 208.0], [1178.0, 208.0], [1178.0, 210.0], [1179.0, 210.0], [1179.0, 211.0], [1180.0, 211.0], [1180.0, 212.0], [1181.0, 212.0], [1181.0, 213.0], [1182.0, 213.0], [1182.0, 214.0], [1183.0, 214.0], [1183.0, 215.0], [1184.0, 215.0], [1184.0, 216.0], [1185.0, 216.0], [1185.0, 217.0], [1187.0, 217.0], [1187.0, 218.0], [1188.0, 218.0], [1188.0, 219.0], [1189.0, 219.0], [1189.0, 220.0], [1190.0, 220.0], [1190.0, 221.0], [1193.0, 221.0], [1193.0, 222.0], [1196.0, 222.0], [1196.0, 223.0], [1197.0, 223.0], [1197.0, 224.0], [1198.0, 224.0], [1198.0, 227.0], [1199.0, 227.0], [1199.0, 238.0], [1200.0, 238.0], [1200.0, 245.0], [1201.0, 245.0], [1201.0, 248.0], [1202.0, 248.0], [1202.0, 249.0], [1203.0, 249.0], [1203.0, 251.0], [1204.0, 251.0], [1204.0, 255.0], [1205.0, 255.0], [1205.0, 261.0], [1206.0, 261.0], [1206.0, 265.0], [1207.0, 265.0], [1207.0, 268.0], [1208.0, 268.0], [1208.0, 270.0], [1209.0, 270.0], [1209.0, 273.0], [1210.0, 273.0], [1210.0, 279.0], [1211.0, 279.0], [1211.0, 378.0], [1210.0, 378.0], [1210.0, 392.0], [1209.0, 392.0], [1209.0, 428.0], [1208.0, 428.0], [1208.0, 437.0], [1207.0, 437.0], [1207.0, 440.0], [1208.0, 440.0], [1208.0, 451.0], [1209.0, 451.0], [1209.0, 454.0], [1210.0, 454.0], [1210.0, 458.0], [1211.0, 458.0], [1211.0, 461.0], [1212.0, 461.0], [1212.0, 463.0], [1213.0, 463.0], [1213.0, 464.0], [1214.0, 464.0], [1214.0, 465.0], [1215.0, 465.0], [1215.0, 466.0], [1216.0, 466.0], [1216.0, 467.0], [1217.0, 467.0], [1217.0, 469.0], [1218.0, 469.0], [1218.0, 470.0], [1220.0, 470.0], [1220.0, 471.0], [1221.0, 471.0], [1221.0, 472.0], [1222.0, 472.0], [1222.0, 474.0], [1223.0, 474.0], [1223.0, 475.0], [1224.0, 475.0], [1224.0, 476.0], [1225.0, 476.0], [1225.0, 477.0], [1227.0, 477.0], [1227.0, 479.0], [1228.0, 479.0], [1228.0, 480.0], [1229.0, 480.0], [1229.0, 481.0], [1230.0, 481.0], [1230.0, 482.0], [1231.0, 482.0], [1231.0, 483.0], [1232.0, 483.0], [1232.0, 484.0], [1233.0, 484.0], [1233.0, 485.0], [1234.0, 485.0], [1234.0, 487.0], [1236.0, 487.0], [1236.0, 488.0], [1237.0, 488.0], [1237.0, 489.0], [1238.0, 489.0], [1238.0, 490.0], [1239.0, 490.0], [1239.0, 491.0], [1240.0, 491.0], [1240.0, 492.0], [1241.0, 492.0], [1241.0, 493.0], [1242.0, 493.0], [1242.0, 494.0], [1244.0, 494.0], [1244.0, 495.0], [1245.0, 495.0], [1245.0, 497.0], [1246.0, 497.0], [1246.0, 498.0], [1248.0, 498.0], [1248.0, 499.0], [1249.0, 499.0], [1249.0, 500.0], [1250.0, 500.0], [1250.0, 501.0], [1251.0, 501.0], [1251.0, 502.0], [1252.0, 502.0], [1252.0, 503.0], [1253.0, 503.0], [1253.0, 504.0], [1255.0, 504.0], [1255.0, 505.0], [1256.0, 505.0], [1256.0, 506.0], [1257.0, 506.0], [1257.0, 508.0], [1258.0, 508.0], [1258.0, 509.0], [1259.0, 509.0], [1259.0, 510.0], [1261.0, 510.0], [1261.0, 511.0], [1262.0, 511.0], [1262.0, 513.0], [1263.0, 513.0], [1263.0, 514.0], [1264.0, 514.0], [1264.0, 515.0], [1265.0, 515.0], [1265.0, 516.0], [1267.0, 516.0], [1267.0, 517.0], [1268.0, 517.0], [1268.0, 519.0], [1269.0, 519.0], [1269.0, 520.0], [1270.0, 520.0], [1270.0, 521.0], [1272.0, 521.0], [1272.0, 522.0], [1273.0, 522.0], [1273.0, 523.0], [1274.0, 523.0], [1274.0, 525.0], [1275.0, 525.0], [1275.0, 526.0], [1276.0, 526.0], [1276.0, 527.0], [1278.0, 527.0], [1278.0, 528.0], [1279.0, 528.0], [1279.0, 530.0], [1280.0, 530.0], [1280.0, 531.0], [1281.0, 531.0], [1281.0, 532.0], [1282.0, 532.0], [1282.0, 533.0], [1284.0, 533.0], [1284.0, 535.0], [1285.0, 535.0], [1285.0, 536.0], [1286.0, 536.0], [1286.0, 537.0], [1287.0, 537.0], [1287.0, 538.0], [1288.0, 538.0], [1288.0, 539.0], [1290.0, 539.0], [1290.0, 540.0], [1291.0, 540.0], [1291.0, 541.0], [1292.0, 541.0], [1292.0, 543.0], [1294.0, 543.0], [1294.0, 544.0], [1295.0, 544.0], [1295.0, 545.0], [1296.0, 545.0], [1296.0, 546.0], [1298.0, 546.0], [1298.0, 548.0], [1300.0, 548.0], [1300.0, 549.0], [1301.0, 549.0], [1301.0, 550.0], [1302.0, 550.0], [1302.0, 551.0], [1303.0, 551.0], [1303.0, 552.0], [1304.0, 552.0], [1304.0, 553.0], [1305.0, 553.0], [1305.0, 555.0], [1306.0, 555.0], [1306.0, 556.0], [1307.0, 556.0], [1307.0, 558.0], [1308.0, 558.0], [1308.0, 560.0], [1309.0, 560.0], [1309.0, 561.0], [1310.0, 561.0], [1310.0, 562.0], [1311.0, 562.0], [1311.0, 563.0], [1312.0, 563.0], [1312.0, 565.0], [1313.0, 565.0], [1313.0, 571.0], [1314.0, 571.0], [1314.0, 573.0], [1315.0, 573.0], [1315.0, 574.0], [1316.0, 574.0], [1316.0, 575.0], [1317.0, 575.0], [1317.0, 577.0], [1318.0, 577.0], [1318.0, 578.0], [1455.0, 578.0], [1455.0, 0.0], [1446.0, 0.0], [1446.0, 3.0], [1440.0, 3.0], [1440.0, 4.0], [1435.0, 4.0], [1435.0, 5.0], [1434.0, 5.0], [1434.0, 4.0], [1425.0, 4.0], [1425.0, 5.0], [1422.0, 5.0], [1422.0, 4.0], [1363.0, 4.0], [1363.0, 3.0], [1326.0, 3.0], [1326.0, 4.0], [1295.0, 4.0], [1295.0, 3.0], [1291.0, 3.0], [1291.0, 4.0], [1283.0, 4.0], [1283.0, 3.0], [1280.0, 3.0], [1280.0, 4.0], [1265.0, 4.0], [1265.0, 5.0], [1263.0, 5.0], [1263.0, 4.0], [1222.0, 4.0], [1222.0, 5.0], [1169.0, 5.0], [1169.0, 4.0], [1157.0, 4.0], [1157.0, 3.0], [1150.0, 3.0], [1150.0, 0.0], [1130.0, 0.0], [1130.0, 3.0], [1119.0, 3.0], [1119.0, 4.0], [1052.0, 4.0], [1052.0, 5.0], [1047.0, 5.0], [1047.0, 4.0], [1032.0, 4.0], [1032.0, 3.0], [1022.0, 3.0], [1022.0, 0.0], [1015.0, 0.0], [1015.0, 3.0], [1014.0, 3.0], [1014.0, 0.0], [849.0, 0.0], [849.0, 3.0], [837.0, 3.0], [837.0, 4.0], [792.0, 4.0], [792.0, 5.0], [773.0, 5.0], [773.0, 4.0], [742.0, 4.0], [742.0, 5.0], [728.0, 5.0], [728.0, 4.0], [701.0, 4.0], [701.0, 3.0], [686.0, 3.0], [686.0, 0.0], [666.0, 0.0], [666.0, 3.0], [651.0, 3.0], [651.0, 4.0], [650.0, 4.0], [650.0, 3.0], [643.0, 3.0], [643.0, 4.0], [638.0, 4.0], [638.0, 3.0], [634.0, 3.0], [634.0, 0.0], [617.0, 0.0], [617.0, 3.0], [606.0, 3.0], [606.0, 0.0], [599.0, 0.0]]}]

        """
        self.sam.prepare_prompt_process(self.source, device=self.device, retina_masks=self.retina_mask,
                                        imgsz=self.imgsz, conf=self.conf,
                                        iou=self.iou)




