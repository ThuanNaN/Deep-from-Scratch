from typing import Any, Dict, List, Tuple
import  torch
from torch import nn, Tensor
from torch.nn import functional as F
from .image_encoder import ImageEncoderViT
from .prompt_encoder import PromptEncoder
from .mask_decoder import MaskDecoder




class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(self, 
                 image_encoder: ImageEncoderViT,
                 prompt_encoder: PromptEncoder,
                 mask_decoder: MaskDecoder,
                 pixel_mean: List[float] = [123.675, 116.28, 103.53],
                 pixel_std: List[float] = [58.395, 57.12, 57.375],
                 ) -> None:
        """
        Args:
            image_encoder (ImageEncoderViT):
            prompt_encoder (PromptEncoder):
            mask_decoder (MaskDecoder):
            pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
            pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(self, 
                batched_input: List[Dict[str, Any]],
                multitask_output: bool
                )-> List[Dict[str, Tensor]]:
        """
        
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeds = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embed in zip(batched_input, image_embeds):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeds, dense_embeds = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_preds = self.mask_decoder(
                image_embeds=curr_embed.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeds=sparse_embeds,
                dense_embeds=dense_embeds,
                multitask_output=multitask_output,
            )
            masks = self.postprocess(
                low_res_masks,
                input_size = image_record["image"].shape[-2:],
                original_size = image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_preds,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(self,
                    masks: Tensor,
                    input_size: Tuple[int, ...],
                    original_size: Tuple[int, ...],
                    ) -> Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """  
        img_sz = self.image_encoder.img_size
        masks = F.interpolate(
            masks,
            (img_sz, img_sz),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., :input_size[0], :input_size[1]]  
        masks = F.interpolate(
            masks,
            original_size,
            mode="bilinear",
            align_corners=False,
        )
        return masks


    def preprocess(self, x: Tensor)->Tensor:
        """
        Normalize pixel values and pad to a square input.
        """
        x = (x - self.pixel_mean) / self.pixel_std

        h, w = x.shape[-2:]
        pad_h = self.image_encoder.img_size - h
        pad_w = self.image_encoder.img_size - w

        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x