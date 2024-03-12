from typing import Type, Tuple, Optional, Any
import numpy as np  
import torch
from torch import nn, Tensor
from .common import LayerNorm2d


class PromptEncoder(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                 img_embed_size: Tuple[int, int],
                 input_img_size: Tuple[int, int],
                 mask_in_channels: int,
                 activation: Type[nn.Module] = nn.GELU,
                 ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Args:
            embed_dim (int): The prompts' embedding dimension 
            img_embed_size (tuple(int, int)): The spatial size of the image embedding, as (H, W).
            input_img_size (int): The padded size of the image as input to the image encoder, as (H, W).
            mask_in_channels (int): The number of hidden channels used for encoding input masks.
            activation (nn.Module): The activation to use when encoding input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_img_size = input_img_size
        self.img_embed_size = img_embed_size

        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeds: int = 4   # pos/neg point + 2 box corners
        point_embeds = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeds)]
        self.point_embeds = nn.ModuleList(point_embeds)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * img_embed_size[0] * img_embed_size[1],)
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_channels // 4, kernel_size=2, stride=2), 
            LayerNorm2d(mask_in_channels // 4),
            activation(),
            nn.Conv2d(mask_in_channels // 4, mask_in_channels, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_channels),
            activation(),
            nn.Conv2d(mask_in_channels, embed_dim, kernel_size=1)
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self)->Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.
        """
        return self.pe_layer(self.img_embed_size).unsqueeze(0) 
    
    def _embed_points(self, 
                      points: Tensor,
                      labels: Tensor,
                      pad: bool
                      )->Tensor:
        """
        Embeds point prompts.
        """
        points += 0.5 # shift to center of pixel
        if pad: 
            padding_points = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_labels = torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat((points, padding_points), dim=1)
            labels = torch.cat((labels, padding_labels), dim=1) 

        point_embeds = self.pe_layer.forward_with_coords(points, self.input_img_size)
        point_embeds[labels == -1] = 0.0
        point_embeds[labels == -1] += self.not_a_point_embed.weight
        point_embeds[labels == 0] += self.point_embeds[0].weight
        point_embeds[labels == 1] += self.point_embeds[1].weight
        return point_embeds

    def _embed_boxes(self,
                     boxes: Tensor
                    )->Tensor:
        """
        Embeds box prompts.
        """
        boxes += 0.5 # shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embeds = self.pe_layer.forward_with_coords(coords, self.input_img_size)
        corner_embeds[:, 0, :] += self.point_embeds[2].weight
        corner_embeds[:, 1, :] += self.point_embeds[3].weight   
        return corner_embeds

    def _embed_masks(self,
                     masks: Tensor
                     )->Tensor:
        """
        Embeds mask prompts.
        """
        return self.mask_downscaling(masks)
    
    def _get_batch_size(self, 
                        points: Tensor,
                        boxes: Tensor,
                        masks: Tensor
                        )->int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes[0].shape[0]
        elif masks is not None:
            return masks[0].shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeds[0].weight.device

    def forward(self,
                points: Optional[Tuple[Tensor, Tensor]],
                boxes: Optional[Tensor],
                masks: Optional[Tensor],
                )->Tuple[Tensor, Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
            points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates and labels to embed.
            boxes (torch.Tensor or none): boxes to embed
            masks (torch.Tensor or none): masks to embed

        Returns:
            torch.Tensor: sparse embeddings for the points and boxes, with shape
                BxNx(embed_dim), where N is determined by the number of input points and boxes.
            torch.Tensor: dense embeddings for the masks, in the shape Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeds = torch.empty((bs, 0, self.embed_dim), device=self._get_device()) 
        if points is not None:
            coords, lables = points
            point_embeds = self._embed_points(coords, lables, pad=(boxes is None))
            sparse_embeds = torch.cat([sparse_embeds, point_embeds], dim=1)
        if boxes is not None:
            box_embeds = self._embed_boxes(boxes)
            sparse_embeds = torch.cat([sparse_embeds, box_embeds], dim=1)
        if masks is not None:
            dense_embeds = self._embed_masks(masks)
        else:
            dense_embeds = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.img_embed_size[0], self.img_embed_size[1]
            )

        return sparse_embeds, dense_embeds
        


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """
    def __init__(self,
                 num_pos_feats: int = 64,
                 scale: Optional[float] = None,
                 ) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix", 
            scale * torch.randn((2, num_pos_feats))
        )
    
    def _pe_encoding(self, coords: Tensor)->Tensor: 
        """
        Positionally encode points that are normalized to [0,1].
        """
        coords = 2*coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
    

    def forward(self, size: Tuple[int, int])->Tensor:
        """
        Generate positional encoding for a grid of the specified size.
        """
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5

        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)
    

    def forward_with_coords(self, coords_input: Tensor, image_size: Tuple[int, int])->Tensor:
        """
        Positionally encode points that are not normalized to [0,1].
        """
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))