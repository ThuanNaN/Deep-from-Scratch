import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np


def main():
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # Inputs
    prompts = "This is a string"
    text_obj = np.array([prompts], dtype="object")
    

    input_tensors = [
        httpclient.InferInput("TEXT", text_obj.shape, np_to_triton_dtype(text_obj.dtype)),
    ]

    input_tensors[0].set_data_from_numpy(text_obj)


    output = [
        httpclient.InferRequestedOutput("input_ids"),
    ]

    # Query
    query_response = client.infer(model_name="tokenizer",
                                  inputs=input_tensors,
                                  outputs=output)

    print(query_response.as_numpy('input_ids'))

if __name__ == "__main__":
    main()