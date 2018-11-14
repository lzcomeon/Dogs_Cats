import torch as t
import time


class BasicModule(t.nn.Module):
    '''
    封装nn.Module,提供快速加载和保存模型的接口
    '''

    def __init__(self):
        super(BasicModule, self).__init__()
        self.modele_name = str(type(self))


    def load(self, path):
        '''可加载指定路径的模型'''
        self.load_state_dict(t.load(path))


    def save(self, name=None):
        '''
        保存模型，默认使用“模型名字+时间“作为文件名
        :param name:
        :return:
        '''
        if name is None:
            prefix = 'checkpoints/' + self.modele_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name