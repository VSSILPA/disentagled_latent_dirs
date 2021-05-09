# disentagled_latent_dirs
Imporved disentagled directions learning

Change indexing from memory  map to full data in MPI3D, Shapes3D

Normalisation of dtatasets : [0,1]
Change to [-1,1] if required


Cars3D dataset can't be normalised for the whole data due to memory erroR.
Change the train function by loading data and normalize it using an IF CONDITION.