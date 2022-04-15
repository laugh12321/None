import json
from collections import defaultdict
from predict import model_load, model_detect


def preprocess(labels: list) -> str:
    """
    预处理函数：将输入数据预处理为字典类型，并以Json格式输出
    """
    result = defaultdict(list)
    result['num'] = int(len(labels))  
    for label in labels:
        dic = dict()
        dic['Label'], dic['ID'], dic['outline'] = int(label[0]), int(label[1]), label[2]
        result['content'].append(dic)

    return json.dumps(result)


class Model_TransUNet(object):
    def __init__(self, num_classes: int, img_size: int = 256):
        super(Model_TransUNet, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size

    def restore(self, weights):
        self.model = model_load(model_path=weights, 
                                num_classes=self.num_classes, 
                                img_size=self.img_size)

    def redetect(self, file_path, noviz=True):
        labels = model_detect(self.model, 
                              file_path, 
                              noviz=noviz, 
                              num_classes=self.num_classes, 
                              img_size=self.img_size)

        return preprocess(labels)


def initialization():
    global model_unet_irrsign, model_unet_crack

    checkpoints_crack_dir = './checkpoints/unet_crack.pth'
    checkpoints_irrsign_dir = './checkpoints/unet_irrsign.pth'

    print("Loading model...")

    model_unet_irrsign = Model_TransUNet(num_classes=5)
    model_unet_irrsign.restore(weights=checkpoints_irrsign_dir)

    model_unet_crack = Model_TransUNet(num_classes=4)
    model_unet_crack.restore(weights=checkpoints_crack_dir)

    print("Model is ready!")


# 道路病害检测
def detect_unet_crack(file_path, noviz=False):
    return model_unet_crack.redetect(file_path=file_path, noviz=noviz)


if __name__ == '__main__':
    initialization()
    img_dir = './image/road_crack.jpg'

    print(detect_unet_crack(file_path=img_dir, noviz=False))