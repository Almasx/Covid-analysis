from tensorboard import program
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', 'covid19_logs'])
url = tb.launch()