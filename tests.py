from models.ViT import ViT_Base


img_size = 224


model = ViT_Base(
    image_size=img_size,
    patch_size=32,
    num_classes=2,
    dim = 1024,
    depth=6,
    heads=8,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)



print(model)


