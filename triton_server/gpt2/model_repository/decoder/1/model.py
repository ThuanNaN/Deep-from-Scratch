import time
import numpy as np
from typing import Dict, List
import tiktoken
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, PreTrainedTokenizer, TensorType


class TritonPythonModel:
    tokenizer: PreTrainedTokenizer

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

    def execute(self, requests) :
    # -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            output_ids = pb_utils.get_input_tensor_by_name(request, "output_ids").as_numpy().tolist()
            text = self.tokenizer.decode(output_ids)
            text_obj = np.array([text], dtype="object")
            tensor_output = pb_utils.Tensor("TEXT_OUT", text_obj)
 
            inference_response = pb_utils.InferenceResponse(output_tensors = [tensor_output])
            responses.append(inference_response)
        
        # print("CHECK: ", type(responses), len(responses))
        return responses


    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')