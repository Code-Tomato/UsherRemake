# Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps
mixes=(
"vgg11-1-40-134037-0.25-0.25-0.25-744.65" # 25% Job Mix Plan (GPU 3)
"diffusion_1024_1024-1-51-29-0.5-0.5-0.25-0.0923146137935218 vgg19-1-41-30517-0.5-0.5-0.25-169.54" # 50% Job Mix Plan (GPU 3)
"diffusion_1024_1024-1-51-29-0.75-1.25-0.25-0.0923146137935218 vgg19-1-41-30517-0.75-1.25-0.25-169.54" # 75% and 100% Job Mix Plan (GPU 3)
"vgg11-1-40-134037-0.75-1.25-0.25-744.65" # 75% and 100% Job Mix Plan (GPU 4)
)