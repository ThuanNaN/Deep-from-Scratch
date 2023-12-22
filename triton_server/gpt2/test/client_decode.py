import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np


def main():
    client = httpclient.InferenceServerClient(url="localhost:8000")

    ids = list([ 1212 ,  318 ,  257 , 4731])
    output_ids = np.array(ids)
    

    input_tensors = [
        httpclient.InferInput("output_ids", output_ids.shape, np_to_triton_dtype(output_ids.dtype)),
    ]

    input_tensors[0].set_data_from_numpy(output_ids)


    output = [
        httpclient.InferRequestedOutput("TEXT_OUT"),
    ]

    # Query
    query_response = client.infer(model_name="decoder",
                                  inputs=input_tensors,
                                  outputs=output)

    print(query_response.as_numpy('TEXT_OUT')[0].decode("UTF-8"))

if __name__ == "__main__":
    main()