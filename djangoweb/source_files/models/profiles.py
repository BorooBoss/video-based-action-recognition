# max_new_tokens (64-256) max tokens; 1 token = 0,75 word
# do_sample (true/false) true - more creative; false = most prob.
# temperature (0.1 - 2.0)     accuracy     0,2-0,4 / 0,6-0,7 / 0,8 - 1         creative + possible randoms
# top_p   safety / diverse
# repetition_penalty (1.0-2.0)  1.0 / 1.2 / 1.3 -1.5
# no_repeat_ngram_size (2-8) repeating n tokens
# length_penalty (0.5-2.0, default 1.0)

#profiles for paligemma
GENERATION_CONFIGS = {
    "detect": {
        "max_new_tokens": 64,
        "do_sample": False,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3
    },

    "text_generation":{
        "max_new_tokens": 200,
        "do_sample": False,              
        "num_beams": 4,                  
        "early_stopping": True,
        "repetition_penalty": 1.3,
        "no_repeat_ngram_size": 4,
        "length_penalty": 1.1,
    }

}