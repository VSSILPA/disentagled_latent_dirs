def get_data_loader(config, opt):
    if opt.dataset == 'dsprites':
        from data.dsprites import DSprites
        data = DSprites(config, opt)
        return data
    elif opt.dataset == 'mpi3d':
        from data.mpi3d import MPI3D
        data = MPI3D(config, opt)
        return data
    elif opt.dataset == 'shapes3d':
        from data.shapes3d import Shapes3d
        data = Shapes3d(config, opt)
        return data
    elif opt.dataset == 'cars3d':
        from data.cars3d import Cars3D
        data = Cars3D(config, opt)
        return data
    elif opt.dataset == 'celebA':
        raise NotImplementedError
    elif opt.dataset == 'CelebAHQ':
        opt.image_size = 1024
        opt.num_channels = 3
        return None
    else:
        raise NotImplementedError
