
module KnetLayers

using Knet
using UnicodePlots
using Random
using Distributed

import Base.+, Base.*, Base.==

export NeuralNet, NullLayer, LinearLayer, NonlinearLayer, PoolLayer, ConvLayer
export DropoutLayer, BatchnormLayer, ActivationLayer, BiasLayer, DeconvLayer
export Layer, Dense, Pool, Conv1D, Conv2D, Conv3D, Bias, Conv, Deconv
export ReLU, Sigm, Tanh, Dropout, Batchnorm
export operator, weights, repflow, randlayer, depth, train
export isvalid, outdims, flops, nparams, dimflow

#
# LAYER DEFINITIONS AND CONVENIENCES
#

abstract type Layer end
abstract type NullLayer <: Layer end
abstract type ReductionLayer <: Layer end

# TODO: Just const NeuralNet = Array{Layer}  ?
# no, in future want other topologies than just linear, so will need to
# implement trees or graph to hold Layer objects at vertices
mutable struct NeuralNet  # TODO: make some of these types immutable?
    #indims::NTuple{4,Int}
    layers::Array{Layer}
end
NeuralNet() = NeuralNet(Layer[LinearLayer(1)])

depth(N::NeuralNet) = length(N.layers)

#
# Fully Connected (Linear) Layers
#
mutable struct LinearLayer <: Layer
    size::Int
end
# mat(x) = reshape(x, (X1*X2*...*X[D-1],XD))
operator(L::LinearLayer) = (x, w, b) -> w*mat(x) .+ b   # Fully connected layer
nparams(L::LinearLayer) = 2
nparams(L::LinearLayer, indims) =  L.size * prod(indims[1:end-1])
string(L::LinearLayer) = "Dense($(L.size))"
outdims(L::LinearLayer, indims) = (L.size, indims[end])
weights(L::LinearLayer, indims) = (xavier(L.size, prod(indims[1:end-1])),zeros(L.size))
flops(L::LinearLayer, indims) = prod(indims[1:end-1]).^2*L.size + L.size
==(a::LinearLayer, b::LinearLayer) = a.size == b.size
const Linear = LinearLayer
const Dense = LinearLayer

#
#  Nonlinear (Activation) Layers
#
mutable struct NonlinearLayer <: Layer
    op::Function
end
NonlinearOpList = Any[relu, sigm, tanh]
NonlinearOpNames = Dict{Function,String}(relu=>"ReLU",sigm=>"Sigm",tanh=>"Tanh")
ReLU() = NonlinearLayer(relu)
Sigm() = NonlinearLayer(sigm)
Tanh() = NonlinearLayer(tanh)
function leaky_relu(x, alpha=0.2)
    pos = max(0,x)
    neg = min(0,x) * alpha
    return pos + neg
end
LeakyReLU() = NonlinearLayer(leaky_relu)  # TODO
operator(L::NonlinearLayer) = (x) -> (L.op).(x)
nparams(L::NonlinearLayer) = 0
nparams(L::NonlinearLayer, indims) = 0
string(L::NonlinearLayer) = NonlinearOpNames[L.op]*"()"
outdims(L::NonlinearLayer, indims) = indims
weights(L::NonlinearLayer, indims) = ()
flops(L::NonlinearLayer, indims)  = prod(indims)
==(a::NonlinearLayer, b::NonlinearLayer) = a.op == b.op
const ActivationLayer = NonlinearLayer

#
# Convolution Layers
#
mutable struct ConvLayer <: ReductionLayer
    window::Tuple{Int,Int}
    outchan::Int
    stride::Int
    padding::Int
end
ConvLayer(window::Tuple{Int,Int}, outchan) =
    ConvLayer(window::Tuple{Int,Int}, outchan, 1, 0)  # default: stride=1, padding=0
ConvLayer(w::Int, outchan) = ConvLayer((w,w), outchan)
ConvLayer(w::Int, outchan, stride, padding) = ConvLayer((w,w), outchan, stride, padding)
operator(L::ConvLayer) = (x, w, b)-> conv4(w, x, padding=L.padding, stride=L.stride) .+ b
nparams(L::ConvLayer) = 2   # TODO: rename this to nweights?
nparams(L::ConvLayer, indims) = prod(L.window)* indims[3] * L.outchan
string(L::ConvLayer) = "Conv$(length(L.window))D($(L.window), $(L.outchan), $(L.stride), $(L.padding))"
function outdims(L::ConvLayer, X)
    Y1 = 1 + floor(Int, (X[1] + 2*L.padding - L.window[1]) / L.stride)
    Y2 = 1 + floor(Int, (X[2] + 2*L.padding - L.window[2]) / L.stride)
    O = L.outchan
    N = X[3]
    (Y1, Y2, O, N)
end
weights(L::ConvLayer, indims) = (xavier(L.window[1], L.window[2], indims[3], L.outchan),zeros(1,1,L.outchan,1)) # TODO: remove trailing singleton in bias weights?
function flops(L::ConvLayer, indims)
    mw = prod(L.window)
    mw = mw + mw - 1
    return prod(indims[1:2])*mw*L.outchan
end
==(a::ConvLayer, b::ConvLayer) = (a.window==b.window) && (a.outchan==b.outchan) && (a.stride==b.stride) && (a.padding==b.padding)
const Conv = ConvLayer
const Conv1D = ConvLayer
const Conv2D = ConvLayer
const Conv3D = ConvLayer

#
#  Pooling Layers
#
mutable struct PoolLayer <: ReductionLayer
    window::Int
    padding::Int
    stride::Int
    mode::Int
end
PoolLayer() = PoolLayer(2, 0, 2, 0)
operator(L::PoolLayer) = (x) -> pool(x, window=L.window, padding=L.padding, stride=L.stride, mode=L.mode)
nparams(L::PoolLayer) = 0
nparams(L::PoolLayer, indims) = 0
string(L::PoolLayer) = "Pool($(L.window), $(L.padding), $(L.stride), $(L.mode))"
function outdims(L::PoolLayer, X)
    Y1 = 1 + floor(Int, (X[1] + 2*L.padding - L.window) / L.stride)
    Y2 = 1 + floor(Int, (X[2] + 2*L.padding - L.window) / L.stride)
    (Y1, Y2, X[3], X[4])
end
weights(L::PoolLayer, indims) = ()
flops(L::PoolLayer, indims) = prod(indims) # TODO
==(a::PoolLayer, b::PoolLayer) = (a.window==b.window) && (a.mode==b.mode) && (a.stride==b.stride) && (a.padding==b.padding)
const Pool = PoolLayer

#
# Dropout Layers
#
mutable struct DropoutLayer <: Layer
    prob::Float64
end
DropoutLayer() = DropoutLayer(0.1)
operator(L::DropoutLayer) = (x) -> dropout(x, L.prob)
nparams(L::DropoutLayer) = 0
nparams(L::DropoutLayer, indims) = 0
string(L::DropoutLayer) = "Dropout($(L.prob))"
outdims(L::DropoutLayer, indims) = indims
weights(L::DropoutLayer, indims) = ()
flops(l::DropoutLayer, indims) = prod(indims) # TODO
==(a::DropoutLayer, b::DropoutLayer) = a.prob == b.prob
const Dropout = DropoutLayer

#
# Batchnorm Layers
#
mutable struct BatchnormLayer <: Layer
    moments::Knet.BNMoments
end
BatchnormLayer() = BatchnormLayer(bnmoments())
operator(L::BatchnormLayer) = (x, w) -> batchnorm(x, L.moments, w)
nparams(L::BatchnormLayer) = 1
nparams(L::BatchnormLayer, indims) = indims[3]
string(L::BatchnormLayer) = "Batchnorm()"
outdims(L::BatchnormLayer, indims) = indims
weights(L::BatchnormLayer, indims) = (bnparams(indims[3]),)
flops(L::BatchnormLayer, indims) =  prod(indims)  # TODO
==(a::BatchnormLayer, b::BatchnormLayer) = true
const Batchnorm = BatchnormLayer

#
# Bias layer (add vector)
#
struct  BiasLayer <: Layer end
operator(L::BiasLayer) = (x, w) -> x .+ w
nparams(L::BiasLayer) = 1
nparams(L::BiasLayer, indims) =  prod(indims[1:end-1])
string(L::BiasLayer) = "Bias()"
outdims(L::BiasLayer, indims) = indims
weights(L::BiasLayer, indims) = (zeros(indims[1:end-1]...),)
flops(L::BiasLayer, indims) = prod(indims)
==(a::BiasLayer, b::BiasLayer) = true
const Bias = BiasLayer

#
# Deconvolution Layers  (TODO)
#
mutable struct DeconvLayer <: Layer
    window::Tuple{Int,Int}
    outchan::Int
    stride::Int
    padding::Int
end
DeconvLayer(window::Tuple{Int,Int}, outchan) =
    DeconvLayer(window::Tuple{Int,Int}, outchan, 1, 0)  # default: stride=1, padding=0
DeconvLayer(w::Int, outchan) = DeconvLayer((w,w), outchan)
DeconvLayer(w::Int, outchan, stride, padding) = DeconvLayer((w,w), outchan, stride, padding)
operator(L::DeconvLayer) = (x, w, b)-> deconv4(w, x, padding=L.padding, stride=L.stride) .+ b
nparams(L::DeconvLayer) = 2   # TODO: rename this to nweights?
nparams(L::DeconvLayer, indims) = prod(L.window)* indims[3] * L.outchan
string(L::DeconvLayer) = "Deconv$(length(L.window))D($(L.window), $(L.outchan), $(L.stride), $(L.padding))"
function outdims(L::DeconvLayer, X)
# If w has dimensions (W1,W2,...,O,I) and x has
  # dimensions (X1,X2,...,I,N), the result y will have dimensions (Y1,Y2,...,O,N) where
    Y1 = L.window[1] + L.stride*(X[1]-1)-2L.padding
    Y2 = L.window[2] + L.stride*(X[2]-1)-2L.padding
    O = L.outchan
    N = X[3]
    (Y1, Y2, O, N)
end
weights(L::DeconvLayer, indims) = (xavier(L.window[1], L.window[2], indims[3], L.outchan),zeros(1,1,L.outchan,1)) 
function flops(L::DeconvLayer, indims)
    mw = prod(L.window)
    mw = mw + mw - 1
    return prod(indims[1:2])*mw*L.outchan
end
==(a::DeconvLayer, b::DeconvLayer) = (a.window==b.window) && (a.outchan==b.outchan) && (a.stride==b.stride) && (a.padding==b.padding)
const Deconv = DeconvLayer
const Deconv1D = DeconvLayer
const Deconv2D = DeconvLayer
const Deconv3D = DeconvLayer

#
# NeuralNet and Layer methods
#
operator(A::NeuralNet) = function (w, x)
    i = 1
    for L in A.layers
        n = nparams(L)
        x = operator(L)(x, w[i:i+n-1]...)
        i += n
    end
    return x
end
function weights(net::NeuralNet, indims; atype=Array{Float32})
    #kaiming(h, w, i, o) = sqrt(2/(w*h*o)) .* randn(h, w, i, o)
    W = Any[]
    D = indims  # running value of data size as it passes through net
    for L in net.layers
        push!(W, weights(L, D)...)
        #info(L, " -> ", size(W[end]))
        D = outdims(L, D)
    end
    np = sum([length(w) for w in W])
    return (map(a -> convert(atype, a), W), np)
end

function +(A::Array{Layer,1}, B::Array{Layer,1})
    m = length(A)
    n = length(B)
    C = Array{Layer}(undef, m+n)
    C[1:m] = copy(A)
    C[m+1:m+n] = copy(B)
    return C
end
+(A::NeuralNet, B::NeuralNet) = NeuralNet(A.layers + B.layers)
function *(n::Int, L::Layer)
    A = Array{Layer}(undef, n)
    for i in 1:n
        A[i] = L
    end
    return A
end
function *(n::Int, A::Array{Layer,1})
    m = n*length(A)
    B = Array{Layer}(undef, m)
    for i in 1:m
        @info "$i <- $((i-1)%length(A)+1)"
        B[i] = A[(i-1) % length(A) + 1]
    end
    return B
end
*(n::Int, A::NeuralNet) = NeuralNet(n*A.layers)
function ==(A::NeuralNet, B::NeuralNet)
    if length(A.layers) != length(B.layers)
        return false
    else
        return all(A.layers .== B.layers)
    end
end


string(A::NeuralNet) = "NeuralNet(Layer[" * join([string(x) for x in A.layers], ",") * "])"
Base.show(io::IO, A::NeuralNet) = print(io, string(A))

function Base.show(io::IO, ::MIME"text/plain", A::NeuralNet)
    println(io, "$(depth(A))-layer NeuralNet:")
    for (i, L) in enumerate(A.layers)
        i < depth(A) ? println(io, "  ├ $(string(L)),") : print(io, "  └ $(string(L))")
    end
end

function repflow(A::NeuralNet, indims)
    n = zeros(depth(A)+1)
    Din = indims
    n[1] = prod(Din[1:end-1])
    for (i, L) in enumerate(A.layers)
        Dout = outdims(L, Din)
        n[i+1] = prod(Dout[1:end-1])
        Din = Dout
    end
    return n ./ n[1]
end
function dimflow(A::NeuralNet, indims)
    n = zeros(depth(A)+1)
    D = indims
    @info D
    for L in A.layers
        D = outdims(L, D)
        @info "$L -> $D"
    end
end
function flops(A::NeuralNet, indims::NTuple{4,Int})
    # approximate FLOPS of network
    D = indims
    n = 0
    for (k, L) in enumerate(A.layers)
        n += flops(L, D)
        D = outdims(L, D)
    end
    return n
end

Base.isvalid(dims::NTuple{N,Int}) where N = all(dims .> 0)
Base.isvalid(L::Layer, indims) = isvalid(outdims(L, indims))
function Base.isvalid(A::Array{Layer}, indims)
    dims = indims
    gonelinear = false
    for L in A
        if gonelinear && (typeof(L) == ConvLayer || typeof(L) == PoolLayer ||
            typeof(L) == BatchnormLayer)
            #warn("Conv, Pool, or Batchnorm after Linear is INVALID.")
            return false
        elseif !gonelinear && typeof(L) == LinearLayer
            gonelinear = true
        end
        dims = outdims(L, dims)
        isvalid(dims) || return false
    end
    return true
end
Base.isvalid(A::NeuralNet) = Base.isvalid(A.layers)

function nparams(net::NeuralNet, indims)
    n = 0
    D = indims  # running value of data size as it passes through net
    for L in net.layers
        n += nparams(L, D)
        D = outdims(L, D)
    end
    return n
end


#
# TRAINING AND DATA LOADING HELPERS
#

include("mnist.jl")

xtrn, ytrn, xtst, ytst = mnist()

function train(net::NeuralNet; epochs=10, fast=true, lr=0.001, seed=0xc0ffee,
    optfunc=Adam, ftrain=1.0, atype=KnetArray{Float32})
    Random.seed!(seed)
    gpuid = 0
    if gpu() >= 0 && atype == KnetArray{Float32}
        gpuid = myid() % Knet.gpuCount()
        gpu(gpuid) # activate appropriate GPU
        device="gpu$gpuid"
        atype = KnetArray{Float32}
    else
        device="cpu"
        atype = Array{Float32}
    end
    #infoprefix="$(gethostname())$(myid())$device: "
    @info "received $net"
    batchsize = 100
    Ntrain = round(Int, size(xtrn,4)*ftrain)
    dtrn = minibatch(xtrn[:,:,1,1:Ntrain], ytrn[1:Ntrain], batchsize; xtype=atype)  # keep this on CPU for Augmentor
    dtst = minibatch(xtst, ytst, batchsize; xtype=atype)
    nx, ny, nz = size(xtrn)[1:3]
    indims = (nx, ny, nz, 1)
    (w, np) = weights(net, indims, atype=atype)

    predict = operator(net)
    loss(w, x, ygold) = nll(predict(w, x), ygold)
    lossgradient = grad(loss)
    params = Array{Any}(undef, length(w))
    for k in 1:length(w)
        params[k] = optfunc(lr=lr)
    end

    if !fast 
        @info "epoch 0: $(accuracy(w,dtst,predict))"
    end
    xaug = zeros(Float32, nx, ny, 1, batchsize)
    accvtime = zeros(epochs)
    traintime = time()
    for epoch in 1:epochs
        for (x, y) in dtrn
            #x = xtmp
            x = reshape(x, (nx, ny, nz, batchsize))
            #info("type(x): $(typeof(x)), type(y): $(typeof(y)), type(w): $(typeof(w))")
            #info(size(x))
            g = lossgradient(w, x, y)
            update!(w, g, params)
        end
        if !fast
            accvtime[epoch] = accuracy(w,dtst,predict)
            @info "epoch $epoch: $(accvtime[epoch])"
        end
    end
    if gpu() >= 0
        Knet.cudaDeviceSynchronize()
    end
    traintime = time() - traintime
    testacc = accuracy(w, dtst, predict)
    @info "Test accuracy = $testacc, training time = $traintime"
    if !fast
        @info "Accuracy vs Epoch:"
        println(lineplot(accvtime, color=:green))
    end
    return (testacc, traintime, np)
end
train(A::Array{Layer}; kwargs...) = train(NeuralNet(A); kwargs...)


# Example NeuralNets

mnist_mlp = NeuralNet(Layer[Dense(128), Bias(), ReLU(), Dense(64), Bias(), ReLU(), Dense(10)])

mnist_lenet = NeuralNet(Layer[
        Conv2D(5, 20), Tanh(), Pool(),
        Conv2D(5, 50), Tanh(), Pool(),
        Dense(500), Tanh(),
        Dense(10)
    ])

function test(net::NeuralNet = mnist_lenet; kwargs...)
    #show(STDOUT, "text/plain", net)
    acc, t = train(net; kwargs...)
end


end


if PROGRAM_FILE == "KnetLayers.jl"
    #KnetLayers.dimflow(KnetLayers.mnist_lenet,(28,28,1,1))
    KnetLayers.test(KnetLayers.mnist_mlp, fast=false)
    KnetLayers.test(KnetLayers.mnist_lenet, fast=false)
end
