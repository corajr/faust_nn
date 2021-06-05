using FaustNN

model = chord_model()
train_model(model)
faust_code = gen_faust_code(model)
compile_faust(faust_code, "faust_nn")
