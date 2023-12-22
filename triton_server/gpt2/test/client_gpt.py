import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np


def main():
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # Inputs
    ids = np.array([[1212 , 318 , 257, 4731]])
    # ids = np.array([1212 , 318 , 257, 4731])


    input_tensors = [
        httpclient.InferInput("input_ids", ids.shape, np_to_triton_dtype(ids.dtype)),
    ]

    input_tensors[0].set_data_from_numpy(ids)


    output = [
        httpclient.InferRequestedOutput("output_ids"),
    ]

    # Query
    query_response = client.infer(model_name="gpt2",
                                  inputs=input_tensors,
                                  outputs=output)

    print(query_response.as_numpy('output_ids'))

if __name__ == "__main__":
    main()