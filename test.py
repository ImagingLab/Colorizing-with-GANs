from src import ModelOptions, main

options = ModelOptions().parse()
options.mode = 1
main(options)
