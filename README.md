# disentagled_latent_dirs
Imporved disentagled directions learning

Change indexing from memory  map to full data in MPI3D, Shapes3D

Normalisation of dtatasets : [0,1]
Change to [-1,1] if required


Cars3D dataset can't be normalised for the whole data due to memory erroR.
Change the train function by loading data and normalize it using an IF CONDITION.

config.py
Options for style gan generator , discriminator used for stylegan architecture and not style gan2

shapes3d,mpi3d --- self.images is in range [0,255]
while cars3d,ddsprites in range[0,1]