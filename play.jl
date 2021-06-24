# TODO: figure out how to play Faust code in another process.
# PortAudio doesn't seem to output anything when run as below.
import Distributed

# create an additional process to play sound through.
Distributed.addprocs(1)
wid = Distributed.workers()[1]
c = Distributed.RemoteChannel(()->Channel(1))

Distributed.@everywhere import Faust, PortAudio
Distributed.@everywhere block_size=2048
Distributed.@everywhere samplerate=44100
Distributed.@everywhere function play(c; code="process = 0", block_size=1024)
    d = Faust.compile(code)
    Faust.init!(d)

    # require a device matching the input and output size.
    devices = PortAudio.devices()
    dev = filter(x -> x.maxinchans >= size(d.inputs, 2) && x.maxoutchans == size(d.outputs, 2), devices)[1]
    println("Device: $dev")

    PortAudio.PortAudioStream(dev, dev) do stream
        println("Sample rate: $(stream.samplerate)")
        Faust.init!(d, block_size=block_size, samplerate=Int(stream.samplerate))
        while true
            if isready(c)
                model = take!(c)
                Faust.setparams!(d, Dict(zip(model.param_names, model.params)))
            end
            d.inputs = read(stream, block_size)
            write(stream, Faust.compute!(d))
        end
    end
end

example_code = """
import("stdfaust.lib");

process = (os.oscs(220) * 0.25) <: _, _;
"""

# Distributed.@spawnat wid play(c; code=example_code)
t = @async play(c; code=example_code)