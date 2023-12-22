import os
import gdown
import torch
from typing import Dict, List
from time import perf_counter 
from transformers import GPT2LMHeadModel

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

	def initialize(self, args: Dict[str, str]) -> None:
		"""
		Initialize the tokenization process
		:param args: arguments from Triton config file
		"""

		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		ckpt_path = './student_distill.pt'
		if not os.path.exists(ckpt_path):
			id = "1bzNpJvkPuhxA-qxIkXNP2w9ywYFwaASQ"
			gdown.download(id=id, output=ckpt_path)
		
		ckpt = torch.load(ckpt_path, map_location=self.device )
		self.model = GPT2LMHeadModel(ckpt['model_args'])
		self.model.load_state_dict(ckpt['model'])
		self.model.to(self.device )

	def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
		"""
		Parse and tokenize each request
		:param requests: 1 or more requests received by Triton server.
		:return: text as input tensors
		"""
		input_ids = []
		# attention_mask = []
		# model_inputs = {
		# 			"max_length": np.array([256], dtype=np.int32),
		# 			"min_length": np.array([0], dtype=np.int32),
		# 			"num_beams": np.array([2], dtype=np.int32),
		# 			"num_return_sequences": np.array([1], dtype=np.int32),
		# 			"length_penalty": np.array([1], dtype=np.float32),
		# 			"repetition_penalty": np.array([1.3], dtype=np.float32),
		# 		}
		# t0 = perf_counter()
		for request in requests:
			inp_ids = pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy()
			# attn_m = pb_utils.get_input_tensor_by_name(request, "attention_mask").as_numpy()
			input_ids.append(inp_ids[0])
			# attention_mask.append(attn_m)
		input_ids = np.array(input_ids)
		# input_ids = np.concatenate(input_ids, axis=0)
		# attention_mask = np.concatenate(attention_mask, axis=0)
		# print(input_ids.shape, attention_mask.shape)
		# model_inputs.update({
		# 	"input_ids": input_ids, 
		# 	"attention_mask": attention_mask
		# 	})
		# t1 = perf_counter()			
		input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
		with torch.no_grad():
			outputs = self.model.generate(inputs = input_ids, 
											num_beams=4,
											do_sample=True,
											max_new_tokens=200,
											pad_token_id=50256)

		responses = []
		for output in outputs:
			out_tensor = pb_utils.Tensor("output_ids", np.array(output))
			inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
			responses.append(inference_response)
		
		# t2 = perf_counter()
		# print(t2-t1, t1-t0)
		return responses # [ 1212   318   257  4731    26]
	
	def finalize(self):
		"""`finalize` is called only once when the model is being unloaded.
		Implementing `finalize` function is OPTIONAL. This function allows
		the model to perform any necessary clean ups before exit.
		"""
		print('Cleaning up...')
