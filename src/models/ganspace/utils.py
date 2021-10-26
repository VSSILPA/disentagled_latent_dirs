from models.closedform.closedform_directions import CfLinear, CfOrtho

def load_gs_deformator(opt):
    model_name = opt.algo.ours.model_name
    deformator_type = opt.algo.ours.deformator_type
    if deformator_type == 'linear':
        deformator = CfLinear(opt.algo.ours.latent_dim, opt.algo.ours.num_directions)
    elif deformator_type == 'ortho':
        deformator = CfOrtho(opt.algo.ours.latent_dim, opt.algo.ours.num_directions)
    deformator.cuda()
    return deformator
