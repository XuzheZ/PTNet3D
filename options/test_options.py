from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--datasubdir', type=str, default='test_A', help='subdirectory of datadir containing input files')
        self.parser.add_argument('--whichmodel', type=str, default='ckpt10630080.pth', help='which epoch to load? set to latest to use latest cached model')
        self.isTrain = False
