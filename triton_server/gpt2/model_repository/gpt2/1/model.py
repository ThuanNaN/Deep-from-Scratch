import os
import onnxruntime as ort 
from typing import Dict, List
from time import perf_counter 

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

	def initialize(self, args: Dict[str, str]) -> None:
		"""
		Initialize the tokenization process
		:param args: arguments from Triton config file
		"""
		# more variables in https://github.com/triton-inference-server/python_backend/blob/main/src/python.cc
		path: str = os.path.join(args["model_repository"], args["model_version"], 'checkpoints/model_final.onnx')
		opts = ort.SessionOptions()
		opts.log_severity_level = 4
		providers = ["CUDAExecutionProvider"]
		self.sess = ort.InferenceSession(path, opts, providers=providers)
  
	def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
		"""
		Parse and tokenize each request
		:param requests: 1 or more requests received by Triton server.
		:return: text as input tensors
		"""
		input_ids = []
		attention_mask = []
		ort_inputs = {
					"max_length": np.array([256], dtype=np.int32),
					"min_length": np.array([0], dtype=np.int32),
					"num_beams": np.array([2], dtype=np.int32),
					"num_return_sequences": np.array([1], dtype=np.int32),
					"length_penalty": np.array([1], dtype=np.float32),
					"repetition_penalty": np.array([1.3], dtype=np.float32),
				}
		t0 = perf_counter()
		for request in requests:
			inp_ids = pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy()
			attn_m = pb_utils.get_input_tensor_by_name(request, "attention_mask").as_numpy()
			input_ids.append(inp_ids)
			attention_mask.append(attn_m)
		input_ids = np.concatenate(input_ids, axis=0)
		attention_mask = np.concatenate(attention_mask, axis=0)
		print(input_ids.shape, attention_mask.shape)
		ort_inputs.update({"input_ids": input_ids, "attention_mask": attention_mask})
		t1 = perf_counter()			
		outputs = self.sess.run(None, ort_inputs)[0]

		responses = []
		for output in outputs:
			out_tensor_0 = pb_utils.Tensor("output", output[0])
			inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
			responses.append(inference_response)
		t2 = perf_counter()
		print(t2-t1, t1-t0)
		return responses
	
	def finalize(self):
		"""`finalize` is called only once when the model is being unloaded.
		Implementing `finalize` function is OPTIONAL. This function allows
		the model to perform any necessary clean ups before exit.
		"""
		print('Cleaning up...')
