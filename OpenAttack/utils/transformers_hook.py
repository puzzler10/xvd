class HookCloser:
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
    
    def __call__(self, module, input_, output_):
        if input_[0].shape[1] > 1: 
            self.model_wrapper.curr_embedding_t5 = output_ 
        self.model_wrapper.curr_embedding = output_
        output_.retain_grad()

