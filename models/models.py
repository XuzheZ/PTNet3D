def create_model(opt):
    if opt.dimension.startswith('2'):
        from .PTN_model2D import PTN_local
        from models.networks import define_D
        from models.pre_train_VGG16 import Vgg16

        model = PTN_local(img_size=opt.patch_size)
        ext_discriminator = Vgg16()
        D = define_D(2, 64, 3, 'instance', False, 2, True, [0])
        return model, D, ext_discriminator
    elif opt.dimension.startswith('3'):
        from .PTN_model3D import PTN_local_trans
        from models.networks import define_D_3D as define_D
        from models.pre_r3d_18 import Res3D

        model = PTN_local_trans(img_size=opt.patch_size)
        ext_discriminator = Res3D()
        D = define_D(2, 64, 2, 'instance3D', False, 3, True, [0])
        return model, D, ext_discriminator

    elif opt.model == 'Local_ps':
        from .PTN_model2D import PTN_local
        model = PTN_local()
    else:
        raise NotImplementedError
