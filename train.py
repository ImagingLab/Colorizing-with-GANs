from src import ModelOptions, main

options = ModelOptions().parse()
options.mode = 0
main(options)
