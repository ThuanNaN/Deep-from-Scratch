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
            query = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy().tolist()
            query = query[0].decode("UTF-8")
            tokens = self.tokenizer(text=query, return_tensors=TensorType.NUMPY)
            tokens = {
                k: v.astype(np.int64) for k, v in tokens.items()
            }
            outputs = list()
            for input_name in self.tokenizer.model_input_names:
                tensor_input = pb_utils.Tensor(input_name, tokens[input_name])
                outputs.append(tensor_input)
 
            inference_response = pb_utils.InferenceResponse(output_tensors = outputs)
            responses.append(inference_response)
        
        # print("CHECK: ", type(responses), len(responses))
        return responses



    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')