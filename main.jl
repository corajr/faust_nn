import FaustNN

model = FaustNN.chord_model()
FaustNN.train_model(model)
faust_code = FaustNN.gen_faust_code(model)
FaustNN.compile_faust(faust_code, "faust_nn")
