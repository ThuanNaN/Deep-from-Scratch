import gradio as gr

def process_inputs(text1, text2, image):
    # Process your inputs and feed them to your model here
    # This is a placeholder function, replace with your actual function
    # result = llm_model.predict([text1, text2, image])
    result = "This is a placeholder result"
    return result

iface = gr.Interface(
    fn=process_inputs, 
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter first text here...", label="Thuan"), 
        gr.Textbox(lines=2, placeholder="Enter second text here..."), 
        gr.Image()
    ], 
    outputs="text"
)

iface.launch()