??
?$?#
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
FloorMod
x"T
y"T
z"T"
Ttype:
2	
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle???element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements(
handle???element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??
?
&digit_caps/digit_caps_transform_tensorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*7
shared_name(&digit_caps/digit_caps_transform_tensor
?
:digit_caps/digit_caps_transform_tensor/Read/ReadVariableOpReadVariableOp&digit_caps/digit_caps_transform_tensor*&
_output_shapes
:
*
dtype0
?
 digit_caps/digit_caps_log_priorsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" digit_caps/digit_caps_log_priors
?
4digit_caps/digit_caps_log_priors/Read/ReadVariableOpReadVariableOp digit_caps/digit_caps_log_priors*"
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
%feature_maps/feature_map_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%feature_maps/feature_map_conv1/kernel
?
9feature_maps/feature_map_conv1/kernel/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv1/kernel*&
_output_shapes
: *
dtype0
?
#feature_maps/feature_map_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#feature_maps/feature_map_conv1/bias
?
7feature_maps/feature_map_conv1/bias/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_conv1/bias*
_output_shapes
: *
dtype0
?
$feature_maps/feature_map_norm1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$feature_maps/feature_map_norm1/gamma
?
8feature_maps/feature_map_norm1/gamma/Read/ReadVariableOpReadVariableOp$feature_maps/feature_map_norm1/gamma*
_output_shapes
: *
dtype0
?
#feature_maps/feature_map_norm1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#feature_maps/feature_map_norm1/beta
?
7feature_maps/feature_map_norm1/beta/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_norm1/beta*
_output_shapes
: *
dtype0
?
%feature_maps/feature_map_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*6
shared_name'%feature_maps/feature_map_conv2/kernel
?
9feature_maps/feature_map_conv2/kernel/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv2/kernel*&
_output_shapes
: @*
dtype0
?
#feature_maps/feature_map_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#feature_maps/feature_map_conv2/bias
?
7feature_maps/feature_map_conv2/bias/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_conv2/bias*
_output_shapes
:@*
dtype0
?
$feature_maps/feature_map_norm2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$feature_maps/feature_map_norm2/gamma
?
8feature_maps/feature_map_norm2/gamma/Read/ReadVariableOpReadVariableOp$feature_maps/feature_map_norm2/gamma*
_output_shapes
:@*
dtype0
?
#feature_maps/feature_map_norm2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#feature_maps/feature_map_norm2/beta
?
7feature_maps/feature_map_norm2/beta/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_norm2/beta*
_output_shapes
:@*
dtype0
?
%feature_maps/feature_map_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*6
shared_name'%feature_maps/feature_map_conv3/kernel
?
9feature_maps/feature_map_conv3/kernel/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv3/kernel*&
_output_shapes
:@@*
dtype0
?
#feature_maps/feature_map_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#feature_maps/feature_map_conv3/bias
?
7feature_maps/feature_map_conv3/bias/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_conv3/bias*
_output_shapes
:@*
dtype0
?
$feature_maps/feature_map_norm3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$feature_maps/feature_map_norm3/gamma
?
8feature_maps/feature_map_norm3/gamma/Read/ReadVariableOpReadVariableOp$feature_maps/feature_map_norm3/gamma*
_output_shapes
:@*
dtype0
?
#feature_maps/feature_map_norm3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#feature_maps/feature_map_norm3/beta
?
7feature_maps/feature_map_norm3/beta/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_norm3/beta*
_output_shapes
:@*
dtype0
?
%feature_maps/feature_map_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*6
shared_name'%feature_maps/feature_map_conv4/kernel
?
9feature_maps/feature_map_conv4/kernel/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv4/kernel*'
_output_shapes
:@?*
dtype0
?
#feature_maps/feature_map_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#feature_maps/feature_map_conv4/bias
?
7feature_maps/feature_map_conv4/bias/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_conv4/bias*
_output_shapes	
:?*
dtype0
?
$feature_maps/feature_map_norm4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*5
shared_name&$feature_maps/feature_map_norm4/gamma
?
8feature_maps/feature_map_norm4/gamma/Read/ReadVariableOpReadVariableOp$feature_maps/feature_map_norm4/gamma*
_output_shapes	
:?*
dtype0
?
#feature_maps/feature_map_norm4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#feature_maps/feature_map_norm4/beta
?
7feature_maps/feature_map_norm4/beta/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_norm4/beta*
_output_shapes	
:?*
dtype0
?
*feature_maps/feature_map_norm1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*feature_maps/feature_map_norm1/moving_mean
?
>feature_maps/feature_map_norm1/moving_mean/Read/ReadVariableOpReadVariableOp*feature_maps/feature_map_norm1/moving_mean*
_output_shapes
: *
dtype0
?
.feature_maps/feature_map_norm1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.feature_maps/feature_map_norm1/moving_variance
?
Bfeature_maps/feature_map_norm1/moving_variance/Read/ReadVariableOpReadVariableOp.feature_maps/feature_map_norm1/moving_variance*
_output_shapes
: *
dtype0
?
*feature_maps/feature_map_norm2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*feature_maps/feature_map_norm2/moving_mean
?
>feature_maps/feature_map_norm2/moving_mean/Read/ReadVariableOpReadVariableOp*feature_maps/feature_map_norm2/moving_mean*
_output_shapes
:@*
dtype0
?
.feature_maps/feature_map_norm2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.feature_maps/feature_map_norm2/moving_variance
?
Bfeature_maps/feature_map_norm2/moving_variance/Read/ReadVariableOpReadVariableOp.feature_maps/feature_map_norm2/moving_variance*
_output_shapes
:@*
dtype0
?
*feature_maps/feature_map_norm3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*feature_maps/feature_map_norm3/moving_mean
?
>feature_maps/feature_map_norm3/moving_mean/Read/ReadVariableOpReadVariableOp*feature_maps/feature_map_norm3/moving_mean*
_output_shapes
:@*
dtype0
?
.feature_maps/feature_map_norm3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.feature_maps/feature_map_norm3/moving_variance
?
Bfeature_maps/feature_map_norm3/moving_variance/Read/ReadVariableOpReadVariableOp.feature_maps/feature_map_norm3/moving_variance*
_output_shapes
:@*
dtype0
?
*feature_maps/feature_map_norm4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*feature_maps/feature_map_norm4/moving_mean
?
>feature_maps/feature_map_norm4/moving_mean/Read/ReadVariableOpReadVariableOp*feature_maps/feature_map_norm4/moving_mean*
_output_shapes	
:?*
dtype0
?
.feature_maps/feature_map_norm4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.feature_maps/feature_map_norm4/moving_variance
?
Bfeature_maps/feature_map_norm4/moving_variance/Read/ReadVariableOpReadVariableOp.feature_maps/feature_map_norm4/moving_variance*
_output_shapes	
:?*
dtype0
?
%primary_caps/primary_cap_dconv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		?*6
shared_name'%primary_caps/primary_cap_dconv/kernel
?
9primary_caps/primary_cap_dconv/kernel/Read/ReadVariableOpReadVariableOp%primary_caps/primary_cap_dconv/kernel*'
_output_shapes
:		?*
dtype0
?
#primary_caps/primary_cap_dconv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#primary_caps/primary_cap_dconv/bias
?
7primary_caps/primary_cap_dconv/bias/Read/ReadVariableOpReadVariableOp#primary_caps/primary_cap_dconv/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
-Adam/digit_caps/digit_caps_transform_tensor/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*>
shared_name/-Adam/digit_caps/digit_caps_transform_tensor/m
?
AAdam/digit_caps/digit_caps_transform_tensor/m/Read/ReadVariableOpReadVariableOp-Adam/digit_caps/digit_caps_transform_tensor/m*&
_output_shapes
:
*
dtype0
?
'Adam/digit_caps/digit_caps_log_priors/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*8
shared_name)'Adam/digit_caps/digit_caps_log_priors/m
?
;Adam/digit_caps/digit_caps_log_priors/m/Read/ReadVariableOpReadVariableOp'Adam/digit_caps/digit_caps_log_priors/m*"
_output_shapes
:
*
dtype0
?
,Adam/feature_maps/feature_map_conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,Adam/feature_maps/feature_map_conv1/kernel/m
?
@Adam/feature_maps/feature_map_conv1/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/feature_maps/feature_map_conv1/kernel/m*&
_output_shapes
: *
dtype0
?
*Adam/feature_maps/feature_map_conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/feature_maps/feature_map_conv1/bias/m
?
>Adam/feature_maps/feature_map_conv1/bias/m/Read/ReadVariableOpReadVariableOp*Adam/feature_maps/feature_map_conv1/bias/m*
_output_shapes
: *
dtype0
?
+Adam/feature_maps/feature_map_norm1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/feature_maps/feature_map_norm1/gamma/m
?
?Adam/feature_maps/feature_map_norm1/gamma/m/Read/ReadVariableOpReadVariableOp+Adam/feature_maps/feature_map_norm1/gamma/m*
_output_shapes
: *
dtype0
?
*Adam/feature_maps/feature_map_norm1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/feature_maps/feature_map_norm1/beta/m
?
>Adam/feature_maps/feature_map_norm1/beta/m/Read/ReadVariableOpReadVariableOp*Adam/feature_maps/feature_map_norm1/beta/m*
_output_shapes
: *
dtype0
?
,Adam/feature_maps/feature_map_conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*=
shared_name.,Adam/feature_maps/feature_map_conv2/kernel/m
?
@Adam/feature_maps/feature_map_conv2/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/feature_maps/feature_map_conv2/kernel/m*&
_output_shapes
: @*
dtype0
?
*Adam/feature_maps/feature_map_conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/feature_maps/feature_map_conv2/bias/m
?
>Adam/feature_maps/feature_map_conv2/bias/m/Read/ReadVariableOpReadVariableOp*Adam/feature_maps/feature_map_conv2/bias/m*
_output_shapes
:@*
dtype0
?
+Adam/feature_maps/feature_map_norm2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+Adam/feature_maps/feature_map_norm2/gamma/m
?
?Adam/feature_maps/feature_map_norm2/gamma/m/Read/ReadVariableOpReadVariableOp+Adam/feature_maps/feature_map_norm2/gamma/m*
_output_shapes
:@*
dtype0
?
*Adam/feature_maps/feature_map_norm2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/feature_maps/feature_map_norm2/beta/m
?
>Adam/feature_maps/feature_map_norm2/beta/m/Read/ReadVariableOpReadVariableOp*Adam/feature_maps/feature_map_norm2/beta/m*
_output_shapes
:@*
dtype0
?
,Adam/feature_maps/feature_map_conv3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*=
shared_name.,Adam/feature_maps/feature_map_conv3/kernel/m
?
@Adam/feature_maps/feature_map_conv3/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/feature_maps/feature_map_conv3/kernel/m*&
_output_shapes
:@@*
dtype0
?
*Adam/feature_maps/feature_map_conv3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/feature_maps/feature_map_conv3/bias/m
?
>Adam/feature_maps/feature_map_conv3/bias/m/Read/ReadVariableOpReadVariableOp*Adam/feature_maps/feature_map_conv3/bias/m*
_output_shapes
:@*
dtype0
?
+Adam/feature_maps/feature_map_norm3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+Adam/feature_maps/feature_map_norm3/gamma/m
?
?Adam/feature_maps/feature_map_norm3/gamma/m/Read/ReadVariableOpReadVariableOp+Adam/feature_maps/feature_map_norm3/gamma/m*
_output_shapes
:@*
dtype0
?
*Adam/feature_maps/feature_map_norm3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/feature_maps/feature_map_norm3/beta/m
?
>Adam/feature_maps/feature_map_norm3/beta/m/Read/ReadVariableOpReadVariableOp*Adam/feature_maps/feature_map_norm3/beta/m*
_output_shapes
:@*
dtype0
?
,Adam/feature_maps/feature_map_conv4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*=
shared_name.,Adam/feature_maps/feature_map_conv4/kernel/m
?
@Adam/feature_maps/feature_map_conv4/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/feature_maps/feature_map_conv4/kernel/m*'
_output_shapes
:@?*
dtype0
?
*Adam/feature_maps/feature_map_conv4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*Adam/feature_maps/feature_map_conv4/bias/m
?
>Adam/feature_maps/feature_map_conv4/bias/m/Read/ReadVariableOpReadVariableOp*Adam/feature_maps/feature_map_conv4/bias/m*
_output_shapes	
:?*
dtype0
?
+Adam/feature_maps/feature_map_norm4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*<
shared_name-+Adam/feature_maps/feature_map_norm4/gamma/m
?
?Adam/feature_maps/feature_map_norm4/gamma/m/Read/ReadVariableOpReadVariableOp+Adam/feature_maps/feature_map_norm4/gamma/m*
_output_shapes	
:?*
dtype0
?
*Adam/feature_maps/feature_map_norm4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*Adam/feature_maps/feature_map_norm4/beta/m
?
>Adam/feature_maps/feature_map_norm4/beta/m/Read/ReadVariableOpReadVariableOp*Adam/feature_maps/feature_map_norm4/beta/m*
_output_shapes	
:?*
dtype0
?
,Adam/primary_caps/primary_cap_dconv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		?*=
shared_name.,Adam/primary_caps/primary_cap_dconv/kernel/m
?
@Adam/primary_caps/primary_cap_dconv/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/primary_caps/primary_cap_dconv/kernel/m*'
_output_shapes
:		?*
dtype0
?
*Adam/primary_caps/primary_cap_dconv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*Adam/primary_caps/primary_cap_dconv/bias/m
?
>Adam/primary_caps/primary_cap_dconv/bias/m/Read/ReadVariableOpReadVariableOp*Adam/primary_caps/primary_cap_dconv/bias/m*
_output_shapes	
:?*
dtype0
?
-Adam/digit_caps/digit_caps_transform_tensor/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*>
shared_name/-Adam/digit_caps/digit_caps_transform_tensor/v
?
AAdam/digit_caps/digit_caps_transform_tensor/v/Read/ReadVariableOpReadVariableOp-Adam/digit_caps/digit_caps_transform_tensor/v*&
_output_shapes
:
*
dtype0
?
'Adam/digit_caps/digit_caps_log_priors/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*8
shared_name)'Adam/digit_caps/digit_caps_log_priors/v
?
;Adam/digit_caps/digit_caps_log_priors/v/Read/ReadVariableOpReadVariableOp'Adam/digit_caps/digit_caps_log_priors/v*"
_output_shapes
:
*
dtype0
?
,Adam/feature_maps/feature_map_conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,Adam/feature_maps/feature_map_conv1/kernel/v
?
@Adam/feature_maps/feature_map_conv1/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/feature_maps/feature_map_conv1/kernel/v*&
_output_shapes
: *
dtype0
?
*Adam/feature_maps/feature_map_conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/feature_maps/feature_map_conv1/bias/v
?
>Adam/feature_maps/feature_map_conv1/bias/v/Read/ReadVariableOpReadVariableOp*Adam/feature_maps/feature_map_conv1/bias/v*
_output_shapes
: *
dtype0
?
+Adam/feature_maps/feature_map_norm1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/feature_maps/feature_map_norm1/gamma/v
?
?Adam/feature_maps/feature_map_norm1/gamma/v/Read/ReadVariableOpReadVariableOp+Adam/feature_maps/feature_map_norm1/gamma/v*
_output_shapes
: *
dtype0
?
*Adam/feature_maps/feature_map_norm1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/feature_maps/feature_map_norm1/beta/v
?
>Adam/feature_maps/feature_map_norm1/beta/v/Read/ReadVariableOpReadVariableOp*Adam/feature_maps/feature_map_norm1/beta/v*
_output_shapes
: *
dtype0
?
,Adam/feature_maps/feature_map_conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*=
shared_name.,Adam/feature_maps/feature_map_conv2/kernel/v
?
@Adam/feature_maps/feature_map_conv2/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/feature_maps/feature_map_conv2/kernel/v*&
_output_shapes
: @*
dtype0
?
*Adam/feature_maps/feature_map_conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/feature_maps/feature_map_conv2/bias/v
?
>Adam/feature_maps/feature_map_conv2/bias/v/Read/ReadVariableOpReadVariableOp*Adam/feature_maps/feature_map_conv2/bias/v*
_output_shapes
:@*
dtype0
?
+Adam/feature_maps/feature_map_norm2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+Adam/feature_maps/feature_map_norm2/gamma/v
?
?Adam/feature_maps/feature_map_norm2/gamma/v/Read/ReadVariableOpReadVariableOp+Adam/feature_maps/feature_map_norm2/gamma/v*
_output_shapes
:@*
dtype0
?
*Adam/feature_maps/feature_map_norm2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/feature_maps/feature_map_norm2/beta/v
?
>Adam/feature_maps/feature_map_norm2/beta/v/Read/ReadVariableOpReadVariableOp*Adam/feature_maps/feature_map_norm2/beta/v*
_output_shapes
:@*
dtype0
?
,Adam/feature_maps/feature_map_conv3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*=
shared_name.,Adam/feature_maps/feature_map_conv3/kernel/v
?
@Adam/feature_maps/feature_map_conv3/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/feature_maps/feature_map_conv3/kernel/v*&
_output_shapes
:@@*
dtype0
?
*Adam/feature_maps/feature_map_conv3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/feature_maps/feature_map_conv3/bias/v
?
>Adam/feature_maps/feature_map_conv3/bias/v/Read/ReadVariableOpReadVariableOp*Adam/feature_maps/feature_map_conv3/bias/v*
_output_shapes
:@*
dtype0
?
+Adam/feature_maps/feature_map_norm3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+Adam/feature_maps/feature_map_norm3/gamma/v
?
?Adam/feature_maps/feature_map_norm3/gamma/v/Read/ReadVariableOpReadVariableOp+Adam/feature_maps/feature_map_norm3/gamma/v*
_output_shapes
:@*
dtype0
?
*Adam/feature_maps/feature_map_norm3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/feature_maps/feature_map_norm3/beta/v
?
>Adam/feature_maps/feature_map_norm3/beta/v/Read/ReadVariableOpReadVariableOp*Adam/feature_maps/feature_map_norm3/beta/v*
_output_shapes
:@*
dtype0
?
,Adam/feature_maps/feature_map_conv4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*=
shared_name.,Adam/feature_maps/feature_map_conv4/kernel/v
?
@Adam/feature_maps/feature_map_conv4/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/feature_maps/feature_map_conv4/kernel/v*'
_output_shapes
:@?*
dtype0
?
*Adam/feature_maps/feature_map_conv4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*Adam/feature_maps/feature_map_conv4/bias/v
?
>Adam/feature_maps/feature_map_conv4/bias/v/Read/ReadVariableOpReadVariableOp*Adam/feature_maps/feature_map_conv4/bias/v*
_output_shapes	
:?*
dtype0
?
+Adam/feature_maps/feature_map_norm4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*<
shared_name-+Adam/feature_maps/feature_map_norm4/gamma/v
?
?Adam/feature_maps/feature_map_norm4/gamma/v/Read/ReadVariableOpReadVariableOp+Adam/feature_maps/feature_map_norm4/gamma/v*
_output_shapes	
:?*
dtype0
?
*Adam/feature_maps/feature_map_norm4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*Adam/feature_maps/feature_map_norm4/beta/v
?
>Adam/feature_maps/feature_map_norm4/beta/v/Read/ReadVariableOpReadVariableOp*Adam/feature_maps/feature_map_norm4/beta/v*
_output_shapes	
:?*
dtype0
?
,Adam/primary_caps/primary_cap_dconv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		?*=
shared_name.,Adam/primary_caps/primary_cap_dconv/kernel/v
?
@Adam/primary_caps/primary_cap_dconv/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/primary_caps/primary_cap_dconv/kernel/v*'
_output_shapes
:		?*
dtype0
?
*Adam/primary_caps/primary_cap_dconv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*Adam/primary_caps/primary_cap_dconv/bias/v
?
>Adam/primary_caps/primary_cap_dconv/bias/v/Read/ReadVariableOpReadVariableOp*Adam/primary_caps/primary_cap_dconv/bias/v*
_output_shapes	
:?*
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??>

NoOpNoOp
??
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*Ơ
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*

_init_input_shape* 
?
	conv1
	norm1
	conv2
	norm2
	conv3
	norm3
	conv4
	norm4
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
	dconv
reshape

 squash
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*
?
'digit_caps_transform_tensor
'W
(digit_caps_log_priors
(B

)squash
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses*
?
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses* 
?
6iter

7beta_1

8beta_2
	9decay
:learning_rate'm?(m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?Sm?Tm?'v?(v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?Sv?Tv?*
?
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15
K16
L17
M18
N19
O20
P21
Q22
R23
S24
T25
'26
(27*
?
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15
S16
T17
'18
(19*
* 
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Zserving_default* 
* 
?

;kernel
<bias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses*
?
aaxis
	=gamma
>beta
Kmoving_mean
Lmoving_variance
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses*
?

?kernel
@bias
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses*
?
naxis
	Agamma
Bbeta
Mmoving_mean
Nmoving_variance
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses*
?

Ckernel
Dbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses*
?
{axis
	Egamma
Fbeta
Omoving_mean
Pmoving_variance
|	variables
}trainable_variables
~regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

Gkernel
Hbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
	?axis
	Igamma
Jbeta
Qmoving_mean
Rmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15
K16
L17
M18
N19
O20
P21
Q22
R23*
z
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
?

Skernel
Tbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 

S0
T1*

S0
T1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 
??
VARIABLE_VALUE&digit_caps/digit_caps_transform_tensorKlayer_with_weights-2/digit_caps_transform_tensor/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE digit_caps/digit_caps_log_priorsElayer_with_weights-2/digit_caps_log_priors/.ATTRIBUTES/VARIABLE_VALUE*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 

'0
(1*

'0
(1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%feature_maps/feature_map_conv1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#feature_maps/feature_map_conv1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$feature_maps/feature_map_norm1/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#feature_maps/feature_map_norm1/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%feature_maps/feature_map_conv2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#feature_maps/feature_map_conv2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$feature_maps/feature_map_norm2/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#feature_maps/feature_map_norm2/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%feature_maps/feature_map_conv3/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#feature_maps/feature_map_conv3/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$feature_maps/feature_map_norm3/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#feature_maps/feature_map_norm3/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%feature_maps/feature_map_conv4/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#feature_maps/feature_map_conv4/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$feature_maps/feature_map_norm4/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#feature_maps/feature_map_norm4/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*feature_maps/feature_map_norm1/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.feature_maps/feature_map_norm1/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*feature_maps/feature_map_norm2/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.feature_maps/feature_map_norm2/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*feature_maps/feature_map_norm3/moving_mean'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.feature_maps/feature_map_norm3/moving_variance'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*feature_maps/feature_map_norm4/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.feature_maps/feature_map_norm4/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%primary_caps/primary_cap_dconv/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#primary_caps/primary_cap_dconv/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
<
K0
L1
M2
N3
O4
P5
Q6
R7*
'
0
1
2
3
4*

?0
?1*
* 
* 
* 

;0
<1*

;0
<1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*
* 
* 
* 
 
=0
>1
K2
L3*

=0
>1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*
* 
* 

?0
@1*

?0
@1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses*
* 
* 
* 
 
A0
B1
M2
N3*

A0
B1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*
* 
* 

C0
D1*

C0
D1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*
* 
* 
* 
 
E0
F1
O2
P3*

E0
F1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
|	variables
}trainable_variables
~regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

G0
H1*

G0
H1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
 
I0
J1
Q2
R3*

I0
J1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
<
K0
L1
M2
N3
O4
P5
Q6
R7*
<
0
1
2
3
4
5
6
7*
* 
* 
* 

S0
T1*

S0
T1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 

0
1
 2*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
	
)0* 
* 
* 
* 
* 
* 
* 
* 
* 
<

?total

?count
?	variables
?	keras_api*
M

?total

?count
?
_fn_kwargs
?	variables
?	keras_api*
* 
* 
* 
* 
* 

K0
L1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

M0
N1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

O0
P1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

Q0
R1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
??
VARIABLE_VALUE-Adam/digit_caps/digit_caps_transform_tensor/mglayer_with_weights-2/digit_caps_transform_tensor/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE'Adam/digit_caps/digit_caps_log_priors/malayer_with_weights-2/digit_caps_log_priors/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/feature_maps/feature_map_conv1/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/feature_maps/feature_map_conv1/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE+Adam/feature_maps/feature_map_norm1/gamma/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/feature_maps/feature_map_norm1/beta/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/feature_maps/feature_map_conv2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/feature_maps/feature_map_conv2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE+Adam/feature_maps/feature_map_norm2/gamma/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/feature_maps/feature_map_norm2/beta/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/feature_maps/feature_map_conv3/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/feature_maps/feature_map_conv3/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE+Adam/feature_maps/feature_map_norm3/gamma/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/feature_maps/feature_map_norm3/beta/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/feature_maps/feature_map_conv4/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/feature_maps/feature_map_conv4/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE+Adam/feature_maps/feature_map_norm4/gamma/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/feature_maps/feature_map_norm4/beta/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/primary_caps/primary_cap_dconv/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/primary_caps/primary_cap_dconv/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE-Adam/digit_caps/digit_caps_transform_tensor/vglayer_with_weights-2/digit_caps_transform_tensor/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE'Adam/digit_caps/digit_caps_log_priors/valayer_with_weights-2/digit_caps_log_priors/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/feature_maps/feature_map_conv1/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/feature_maps/feature_map_conv1/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE+Adam/feature_maps/feature_map_norm1/gamma/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/feature_maps/feature_map_norm1/beta/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/feature_maps/feature_map_conv2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/feature_maps/feature_map_conv2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE+Adam/feature_maps/feature_map_norm2/gamma/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/feature_maps/feature_map_norm2/beta/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/feature_maps/feature_map_conv3/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/feature_maps/feature_map_conv3/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE+Adam/feature_maps/feature_map_norm3/gamma/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/feature_maps/feature_map_norm3/beta/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/feature_maps/feature_map_conv4/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/feature_maps/feature_map_conv4/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE+Adam/feature_maps/feature_map_norm4/gamma/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/feature_maps/feature_map_norm4/beta/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/primary_caps/primary_cap_dconv/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/primary_caps/primary_cap_dconv/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_input_imagesPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_images%feature_maps/feature_map_conv1/kernel#feature_maps/feature_map_conv1/bias$feature_maps/feature_map_norm1/gamma#feature_maps/feature_map_norm1/beta*feature_maps/feature_map_norm1/moving_mean.feature_maps/feature_map_norm1/moving_variance%feature_maps/feature_map_conv2/kernel#feature_maps/feature_map_conv2/bias$feature_maps/feature_map_norm2/gamma#feature_maps/feature_map_norm2/beta*feature_maps/feature_map_norm2/moving_mean.feature_maps/feature_map_norm2/moving_variance%feature_maps/feature_map_conv3/kernel#feature_maps/feature_map_conv3/bias$feature_maps/feature_map_norm3/gamma#feature_maps/feature_map_norm3/beta*feature_maps/feature_map_norm3/moving_mean.feature_maps/feature_map_norm3/moving_variance%feature_maps/feature_map_conv4/kernel#feature_maps/feature_map_conv4/bias$feature_maps/feature_map_norm4/gamma#feature_maps/feature_map_norm4/beta*feature_maps/feature_map_norm4/moving_mean.feature_maps/feature_map_norm4/moving_variance%primary_caps/primary_cap_dconv/kernel#primary_caps/primary_cap_dconv/bias&digit_caps/digit_caps_transform_tensorConst digit_caps/digit_caps_log_priors*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_726349
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?&
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename:digit_caps/digit_caps_transform_tensor/Read/ReadVariableOp4digit_caps/digit_caps_log_priors/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp9feature_maps/feature_map_conv1/kernel/Read/ReadVariableOp7feature_maps/feature_map_conv1/bias/Read/ReadVariableOp8feature_maps/feature_map_norm1/gamma/Read/ReadVariableOp7feature_maps/feature_map_norm1/beta/Read/ReadVariableOp9feature_maps/feature_map_conv2/kernel/Read/ReadVariableOp7feature_maps/feature_map_conv2/bias/Read/ReadVariableOp8feature_maps/feature_map_norm2/gamma/Read/ReadVariableOp7feature_maps/feature_map_norm2/beta/Read/ReadVariableOp9feature_maps/feature_map_conv3/kernel/Read/ReadVariableOp7feature_maps/feature_map_conv3/bias/Read/ReadVariableOp8feature_maps/feature_map_norm3/gamma/Read/ReadVariableOp7feature_maps/feature_map_norm3/beta/Read/ReadVariableOp9feature_maps/feature_map_conv4/kernel/Read/ReadVariableOp7feature_maps/feature_map_conv4/bias/Read/ReadVariableOp8feature_maps/feature_map_norm4/gamma/Read/ReadVariableOp7feature_maps/feature_map_norm4/beta/Read/ReadVariableOp>feature_maps/feature_map_norm1/moving_mean/Read/ReadVariableOpBfeature_maps/feature_map_norm1/moving_variance/Read/ReadVariableOp>feature_maps/feature_map_norm2/moving_mean/Read/ReadVariableOpBfeature_maps/feature_map_norm2/moving_variance/Read/ReadVariableOp>feature_maps/feature_map_norm3/moving_mean/Read/ReadVariableOpBfeature_maps/feature_map_norm3/moving_variance/Read/ReadVariableOp>feature_maps/feature_map_norm4/moving_mean/Read/ReadVariableOpBfeature_maps/feature_map_norm4/moving_variance/Read/ReadVariableOp9primary_caps/primary_cap_dconv/kernel/Read/ReadVariableOp7primary_caps/primary_cap_dconv/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpAAdam/digit_caps/digit_caps_transform_tensor/m/Read/ReadVariableOp;Adam/digit_caps/digit_caps_log_priors/m/Read/ReadVariableOp@Adam/feature_maps/feature_map_conv1/kernel/m/Read/ReadVariableOp>Adam/feature_maps/feature_map_conv1/bias/m/Read/ReadVariableOp?Adam/feature_maps/feature_map_norm1/gamma/m/Read/ReadVariableOp>Adam/feature_maps/feature_map_norm1/beta/m/Read/ReadVariableOp@Adam/feature_maps/feature_map_conv2/kernel/m/Read/ReadVariableOp>Adam/feature_maps/feature_map_conv2/bias/m/Read/ReadVariableOp?Adam/feature_maps/feature_map_norm2/gamma/m/Read/ReadVariableOp>Adam/feature_maps/feature_map_norm2/beta/m/Read/ReadVariableOp@Adam/feature_maps/feature_map_conv3/kernel/m/Read/ReadVariableOp>Adam/feature_maps/feature_map_conv3/bias/m/Read/ReadVariableOp?Adam/feature_maps/feature_map_norm3/gamma/m/Read/ReadVariableOp>Adam/feature_maps/feature_map_norm3/beta/m/Read/ReadVariableOp@Adam/feature_maps/feature_map_conv4/kernel/m/Read/ReadVariableOp>Adam/feature_maps/feature_map_conv4/bias/m/Read/ReadVariableOp?Adam/feature_maps/feature_map_norm4/gamma/m/Read/ReadVariableOp>Adam/feature_maps/feature_map_norm4/beta/m/Read/ReadVariableOp@Adam/primary_caps/primary_cap_dconv/kernel/m/Read/ReadVariableOp>Adam/primary_caps/primary_cap_dconv/bias/m/Read/ReadVariableOpAAdam/digit_caps/digit_caps_transform_tensor/v/Read/ReadVariableOp;Adam/digit_caps/digit_caps_log_priors/v/Read/ReadVariableOp@Adam/feature_maps/feature_map_conv1/kernel/v/Read/ReadVariableOp>Adam/feature_maps/feature_map_conv1/bias/v/Read/ReadVariableOp?Adam/feature_maps/feature_map_norm1/gamma/v/Read/ReadVariableOp>Adam/feature_maps/feature_map_norm1/beta/v/Read/ReadVariableOp@Adam/feature_maps/feature_map_conv2/kernel/v/Read/ReadVariableOp>Adam/feature_maps/feature_map_conv2/bias/v/Read/ReadVariableOp?Adam/feature_maps/feature_map_norm2/gamma/v/Read/ReadVariableOp>Adam/feature_maps/feature_map_norm2/beta/v/Read/ReadVariableOp@Adam/feature_maps/feature_map_conv3/kernel/v/Read/ReadVariableOp>Adam/feature_maps/feature_map_conv3/bias/v/Read/ReadVariableOp?Adam/feature_maps/feature_map_norm3/gamma/v/Read/ReadVariableOp>Adam/feature_maps/feature_map_norm3/beta/v/Read/ReadVariableOp@Adam/feature_maps/feature_map_conv4/kernel/v/Read/ReadVariableOp>Adam/feature_maps/feature_map_conv4/bias/v/Read/ReadVariableOp?Adam/feature_maps/feature_map_norm4/gamma/v/Read/ReadVariableOp>Adam/feature_maps/feature_map_norm4/beta/v/Read/ReadVariableOp@Adam/primary_caps/primary_cap_dconv/kernel/v/Read/ReadVariableOp>Adam/primary_caps/primary_cap_dconv/bias/v/Read/ReadVariableOpConst_1*Z
TinS
Q2O	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_727342
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename&digit_caps/digit_caps_transform_tensor digit_caps/digit_caps_log_priors	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate%feature_maps/feature_map_conv1/kernel#feature_maps/feature_map_conv1/bias$feature_maps/feature_map_norm1/gamma#feature_maps/feature_map_norm1/beta%feature_maps/feature_map_conv2/kernel#feature_maps/feature_map_conv2/bias$feature_maps/feature_map_norm2/gamma#feature_maps/feature_map_norm2/beta%feature_maps/feature_map_conv3/kernel#feature_maps/feature_map_conv3/bias$feature_maps/feature_map_norm3/gamma#feature_maps/feature_map_norm3/beta%feature_maps/feature_map_conv4/kernel#feature_maps/feature_map_conv4/bias$feature_maps/feature_map_norm4/gamma#feature_maps/feature_map_norm4/beta*feature_maps/feature_map_norm1/moving_mean.feature_maps/feature_map_norm1/moving_variance*feature_maps/feature_map_norm2/moving_mean.feature_maps/feature_map_norm2/moving_variance*feature_maps/feature_map_norm3/moving_mean.feature_maps/feature_map_norm3/moving_variance*feature_maps/feature_map_norm4/moving_mean.feature_maps/feature_map_norm4/moving_variance%primary_caps/primary_cap_dconv/kernel#primary_caps/primary_cap_dconv/biastotalcounttotal_1count_1-Adam/digit_caps/digit_caps_transform_tensor/m'Adam/digit_caps/digit_caps_log_priors/m,Adam/feature_maps/feature_map_conv1/kernel/m*Adam/feature_maps/feature_map_conv1/bias/m+Adam/feature_maps/feature_map_norm1/gamma/m*Adam/feature_maps/feature_map_norm1/beta/m,Adam/feature_maps/feature_map_conv2/kernel/m*Adam/feature_maps/feature_map_conv2/bias/m+Adam/feature_maps/feature_map_norm2/gamma/m*Adam/feature_maps/feature_map_norm2/beta/m,Adam/feature_maps/feature_map_conv3/kernel/m*Adam/feature_maps/feature_map_conv3/bias/m+Adam/feature_maps/feature_map_norm3/gamma/m*Adam/feature_maps/feature_map_norm3/beta/m,Adam/feature_maps/feature_map_conv4/kernel/m*Adam/feature_maps/feature_map_conv4/bias/m+Adam/feature_maps/feature_map_norm4/gamma/m*Adam/feature_maps/feature_map_norm4/beta/m,Adam/primary_caps/primary_cap_dconv/kernel/m*Adam/primary_caps/primary_cap_dconv/bias/m-Adam/digit_caps/digit_caps_transform_tensor/v'Adam/digit_caps/digit_caps_log_priors/v,Adam/feature_maps/feature_map_conv1/kernel/v*Adam/feature_maps/feature_map_conv1/bias/v+Adam/feature_maps/feature_map_norm1/gamma/v*Adam/feature_maps/feature_map_norm1/beta/v,Adam/feature_maps/feature_map_conv2/kernel/v*Adam/feature_maps/feature_map_conv2/bias/v+Adam/feature_maps/feature_map_norm2/gamma/v*Adam/feature_maps/feature_map_norm2/beta/v,Adam/feature_maps/feature_map_conv3/kernel/v*Adam/feature_maps/feature_map_conv3/bias/v+Adam/feature_maps/feature_map_norm3/gamma/v*Adam/feature_maps/feature_map_norm3/beta/v,Adam/feature_maps/feature_map_conv4/kernel/v*Adam/feature_maps/feature_map_conv4/bias/v+Adam/feature_maps/feature_map_norm4/gamma/v*Adam/feature_maps/feature_map_norm4/beta/v,Adam/primary_caps/primary_cap_dconv/kernel/v*Adam/primary_caps/primary_cap_dconv/bias/v*Y
TinR
P2N*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_727583??
?
?
2Efficient-CapsNet_digit_caps_map_while_cond_724279^
Zefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_while_loop_counterY
Uefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_strided_slice6
2efficient_capsnet_digit_caps_map_while_placeholder8
4efficient_capsnet_digit_caps_map_while_placeholder_1^
Zefficient_capsnet_digit_caps_map_while_less_efficient_capsnet_digit_caps_map_strided_slicev
refficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_while_cond_724279___redundant_placeholder0v
refficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_while_cond_724279___redundant_placeholder13
/efficient_capsnet_digit_caps_map_while_identity
?
+Efficient-CapsNet/digit_caps/map/while/LessLess2efficient_capsnet_digit_caps_map_while_placeholderZefficient_capsnet_digit_caps_map_while_less_efficient_capsnet_digit_caps_map_strided_slice*
T0*
_output_shapes
: ?
-Efficient-CapsNet/digit_caps/map/while/Less_1LessZefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_while_loop_counterUefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_strided_slice*
T0*
_output_shapes
: ?
1Efficient-CapsNet/digit_caps/map/while/LogicalAnd
LogicalAnd1Efficient-CapsNet/digit_caps/map/while/Less_1:z:0/Efficient-CapsNet/digit_caps/map/while/Less:z:0*
_output_shapes
: ?
/Efficient-CapsNet/digit_caps/map/while/IdentityIdentity5Efficient-CapsNet/digit_caps/map/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "k
/efficient_capsnet_digit_caps_map_while_identity8Efficient-CapsNet/digit_caps/map/while/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
?	
?
2__inference_feature_map_norm1_layer_call_fn_726865

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_feature_map_norm1_layer_call_and_return_conditional_losses_724443?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
M__inference_feature_map_norm2_layer_call_and_return_conditional_losses_726945

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?.
?

2Efficient-CapsNet_digit_caps_map_while_body_724280^
Zefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_while_loop_counterY
Uefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_strided_slice6
2efficient_capsnet_digit_caps_map_while_placeholder8
4efficient_capsnet_digit_caps_map_while_placeholder_1]
Yefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_strided_slice_1_0?
?efficient_capsnet_digit_caps_map_while_tensorarrayv2read_tensorlistgetitem_efficient_capsnet_digit_caps_map_tensorarrayunstack_tensorlistfromtensor_0a
Gefficient_capsnet_digit_caps_map_while_matmul_readvariableop_resource_0:
3
/efficient_capsnet_digit_caps_map_while_identity5
1efficient_capsnet_digit_caps_map_while_identity_15
1efficient_capsnet_digit_caps_map_while_identity_25
1efficient_capsnet_digit_caps_map_while_identity_3[
Wefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_strided_slice_1?
?efficient_capsnet_digit_caps_map_while_tensorarrayv2read_tensorlistgetitem_efficient_capsnet_digit_caps_map_tensorarrayunstack_tensorlistfromtensor_
Eefficient_capsnet_digit_caps_map_while_matmul_readvariableop_resource:
??<Efficient-CapsNet/digit_caps/map/while/MatMul/ReadVariableOp?
XEfficient-CapsNet/digit_caps/map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            ?
JEfficient-CapsNet/digit_caps/map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?efficient_capsnet_digit_caps_map_while_tensorarrayv2read_tensorlistgetitem_efficient_capsnet_digit_caps_map_tensorarrayunstack_tensorlistfromtensor_02efficient_capsnet_digit_caps_map_while_placeholderaEfficient-CapsNet/digit_caps/map/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*&
_output_shapes
:
*
element_dtype0?
<Efficient-CapsNet/digit_caps/map/while/MatMul/ReadVariableOpReadVariableOpGefficient_capsnet_digit_caps_map_while_matmul_readvariableop_resource_0*&
_output_shapes
:
*
dtype0?
-Efficient-CapsNet/digit_caps/map/while/MatMulBatchMatMulV2DEfficient-CapsNet/digit_caps/map/while/MatMul/ReadVariableOp:value:0QEfficient-CapsNet/digit_caps/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*&
_output_shapes
:
?
KEfficient-CapsNet/digit_caps/map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem4efficient_capsnet_digit_caps_map_while_placeholder_12efficient_capsnet_digit_caps_map_while_placeholder6Efficient-CapsNet/digit_caps/map/while/MatMul:output:0*
_output_shapes
: *
element_dtype0:???n
,Efficient-CapsNet/digit_caps/map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
*Efficient-CapsNet/digit_caps/map/while/addAddV22efficient_capsnet_digit_caps_map_while_placeholder5Efficient-CapsNet/digit_caps/map/while/add/y:output:0*
T0*
_output_shapes
: p
.Efficient-CapsNet/digit_caps/map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
,Efficient-CapsNet/digit_caps/map/while/add_1AddV2Zefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_while_loop_counter7Efficient-CapsNet/digit_caps/map/while/add_1/y:output:0*
T0*
_output_shapes
: ?
/Efficient-CapsNet/digit_caps/map/while/IdentityIdentity0Efficient-CapsNet/digit_caps/map/while/add_1:z:0,^Efficient-CapsNet/digit_caps/map/while/NoOp*
T0*
_output_shapes
: ?
1Efficient-CapsNet/digit_caps/map/while/Identity_1IdentityUefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_strided_slice,^Efficient-CapsNet/digit_caps/map/while/NoOp*
T0*
_output_shapes
: ?
1Efficient-CapsNet/digit_caps/map/while/Identity_2Identity.Efficient-CapsNet/digit_caps/map/while/add:z:0,^Efficient-CapsNet/digit_caps/map/while/NoOp*
T0*
_output_shapes
: ?
1Efficient-CapsNet/digit_caps/map/while/Identity_3Identity[Efficient-CapsNet/digit_caps/map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0,^Efficient-CapsNet/digit_caps/map/while/NoOp*
T0*
_output_shapes
: :????
+Efficient-CapsNet/digit_caps/map/while/NoOpNoOp=^Efficient-CapsNet/digit_caps/map/while/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
Wefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_strided_slice_1Yefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_strided_slice_1_0"k
/efficient_capsnet_digit_caps_map_while_identity8Efficient-CapsNet/digit_caps/map/while/Identity:output:0"o
1efficient_capsnet_digit_caps_map_while_identity_1:Efficient-CapsNet/digit_caps/map/while/Identity_1:output:0"o
1efficient_capsnet_digit_caps_map_while_identity_2:Efficient-CapsNet/digit_caps/map/while/Identity_2:output:0"o
1efficient_capsnet_digit_caps_map_while_identity_3:Efficient-CapsNet/digit_caps/map/while/Identity_3:output:0"?
Eefficient_capsnet_digit_caps_map_while_matmul_readvariableop_resourceGefficient_capsnet_digit_caps_map_while_matmul_readvariableop_resource_0"?
?efficient_capsnet_digit_caps_map_while_tensorarrayv2read_tensorlistgetitem_efficient_capsnet_digit_caps_map_tensorarrayunstack_tensorlistfromtensor?efficient_capsnet_digit_caps_map_while_tensorarrayv2read_tensorlistgetitem_efficient_capsnet_digit_caps_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : : : 2|
<Efficient-CapsNet/digit_caps/map/while/MatMul/ReadVariableOp<Efficient-CapsNet/digit_caps/map/while/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
H__inference_feature_maps_layer_call_and_return_conditional_losses_726631
input_imagesJ
0feature_map_conv1_conv2d_readvariableop_resource: ?
1feature_map_conv1_biasadd_readvariableop_resource: 7
)feature_map_norm1_readvariableop_resource: 9
+feature_map_norm1_readvariableop_1_resource: H
:feature_map_norm1_fusedbatchnormv3_readvariableop_resource: J
<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource: J
0feature_map_conv2_conv2d_readvariableop_resource: @?
1feature_map_conv2_biasadd_readvariableop_resource:@7
)feature_map_norm2_readvariableop_resource:@9
+feature_map_norm2_readvariableop_1_resource:@H
:feature_map_norm2_fusedbatchnormv3_readvariableop_resource:@J
<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource:@J
0feature_map_conv3_conv2d_readvariableop_resource:@@?
1feature_map_conv3_biasadd_readvariableop_resource:@7
)feature_map_norm3_readvariableop_resource:@9
+feature_map_norm3_readvariableop_1_resource:@H
:feature_map_norm3_fusedbatchnormv3_readvariableop_resource:@J
<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource:@K
0feature_map_conv4_conv2d_readvariableop_resource:@?@
1feature_map_conv4_biasadd_readvariableop_resource:	?8
)feature_map_norm4_readvariableop_resource:	?:
+feature_map_norm4_readvariableop_1_resource:	?I
:feature_map_norm4_fusedbatchnormv3_readvariableop_resource:	?K
<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource:	?
identity??(feature_map_conv1/BiasAdd/ReadVariableOp?'feature_map_conv1/Conv2D/ReadVariableOp?(feature_map_conv2/BiasAdd/ReadVariableOp?'feature_map_conv2/Conv2D/ReadVariableOp?(feature_map_conv3/BiasAdd/ReadVariableOp?'feature_map_conv3/Conv2D/ReadVariableOp?(feature_map_conv4/BiasAdd/ReadVariableOp?'feature_map_conv4/Conv2D/ReadVariableOp? feature_map_norm1/AssignNewValue?"feature_map_norm1/AssignNewValue_1?1feature_map_norm1/FusedBatchNormV3/ReadVariableOp?3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1? feature_map_norm1/ReadVariableOp?"feature_map_norm1/ReadVariableOp_1? feature_map_norm2/AssignNewValue?"feature_map_norm2/AssignNewValue_1?1feature_map_norm2/FusedBatchNormV3/ReadVariableOp?3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1? feature_map_norm2/ReadVariableOp?"feature_map_norm2/ReadVariableOp_1? feature_map_norm3/AssignNewValue?"feature_map_norm3/AssignNewValue_1?1feature_map_norm3/FusedBatchNormV3/ReadVariableOp?3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1? feature_map_norm3/ReadVariableOp?"feature_map_norm3/ReadVariableOp_1? feature_map_norm4/AssignNewValue?"feature_map_norm4/AssignNewValue_1?1feature_map_norm4/FusedBatchNormV3/ReadVariableOp?3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1? feature_map_norm4/ReadVariableOp?"feature_map_norm4/ReadVariableOp_1?
'feature_map_conv1/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
feature_map_conv1/Conv2DConv2Dinput_images/feature_map_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
(feature_map_conv1/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
feature_map_conv1/BiasAddBiasAdd!feature_map_conv1/Conv2D:output:00feature_map_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? |
feature_map_conv1/ReluRelu"feature_map_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
 feature_map_norm1/ReadVariableOpReadVariableOp)feature_map_norm1_readvariableop_resource*
_output_shapes
: *
dtype0?
"feature_map_norm1/ReadVariableOp_1ReadVariableOp+feature_map_norm1_readvariableop_1_resource*
_output_shapes
: *
dtype0?
1feature_map_norm1/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
"feature_map_norm1/FusedBatchNormV3FusedBatchNormV3$feature_map_conv1/Relu:activations:0(feature_map_norm1/ReadVariableOp:value:0*feature_map_norm1/ReadVariableOp_1:value:09feature_map_norm1/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
 feature_map_norm1/AssignNewValueAssignVariableOp:feature_map_norm1_fusedbatchnormv3_readvariableop_resource/feature_map_norm1/FusedBatchNormV3:batch_mean:02^feature_map_norm1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
"feature_map_norm1/AssignNewValue_1AssignVariableOp<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm1/FusedBatchNormV3:batch_variance:04^feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
'feature_map_conv2/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
feature_map_conv2/Conv2DConv2D&feature_map_norm1/FusedBatchNormV3:y:0/feature_map_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
(feature_map_conv2/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
feature_map_conv2/BiasAddBiasAdd!feature_map_conv2/Conv2D:output:00feature_map_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@|
feature_map_conv2/ReluRelu"feature_map_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
 feature_map_norm2/ReadVariableOpReadVariableOp)feature_map_norm2_readvariableop_resource*
_output_shapes
:@*
dtype0?
"feature_map_norm2/ReadVariableOp_1ReadVariableOp+feature_map_norm2_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
1feature_map_norm2/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
"feature_map_norm2/FusedBatchNormV3FusedBatchNormV3$feature_map_conv2/Relu:activations:0(feature_map_norm2/ReadVariableOp:value:0*feature_map_norm2/ReadVariableOp_1:value:09feature_map_norm2/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
 feature_map_norm2/AssignNewValueAssignVariableOp:feature_map_norm2_fusedbatchnormv3_readvariableop_resource/feature_map_norm2/FusedBatchNormV3:batch_mean:02^feature_map_norm2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
"feature_map_norm2/AssignNewValue_1AssignVariableOp<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm2/FusedBatchNormV3:batch_variance:04^feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
'feature_map_conv3/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
feature_map_conv3/Conv2DConv2D&feature_map_norm2/FusedBatchNormV3:y:0/feature_map_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
(feature_map_conv3/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
feature_map_conv3/BiasAddBiasAdd!feature_map_conv3/Conv2D:output:00feature_map_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@|
feature_map_conv3/ReluRelu"feature_map_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
 feature_map_norm3/ReadVariableOpReadVariableOp)feature_map_norm3_readvariableop_resource*
_output_shapes
:@*
dtype0?
"feature_map_norm3/ReadVariableOp_1ReadVariableOp+feature_map_norm3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
1feature_map_norm3/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
"feature_map_norm3/FusedBatchNormV3FusedBatchNormV3$feature_map_conv3/Relu:activations:0(feature_map_norm3/ReadVariableOp:value:0*feature_map_norm3/ReadVariableOp_1:value:09feature_map_norm3/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
 feature_map_norm3/AssignNewValueAssignVariableOp:feature_map_norm3_fusedbatchnormv3_readvariableop_resource/feature_map_norm3/FusedBatchNormV3:batch_mean:02^feature_map_norm3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
"feature_map_norm3/AssignNewValue_1AssignVariableOp<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm3/FusedBatchNormV3:batch_variance:04^feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
'feature_map_conv4/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
feature_map_conv4/Conv2DConv2D&feature_map_norm3/FusedBatchNormV3:y:0/feature_map_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
?
(feature_map_conv4/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
feature_map_conv4/BiasAddBiasAdd!feature_map_conv4/Conv2D:output:00feature_map_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?}
feature_map_conv4/ReluRelu"feature_map_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		??
 feature_map_norm4/ReadVariableOpReadVariableOp)feature_map_norm4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"feature_map_norm4/ReadVariableOp_1ReadVariableOp+feature_map_norm4_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
1feature_map_norm4/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
"feature_map_norm4/FusedBatchNormV3FusedBatchNormV3$feature_map_conv4/Relu:activations:0(feature_map_norm4/ReadVariableOp:value:0*feature_map_norm4/ReadVariableOp_1:value:09feature_map_norm4/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????		?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
 feature_map_norm4/AssignNewValueAssignVariableOp:feature_map_norm4_fusedbatchnormv3_readvariableop_resource/feature_map_norm4/FusedBatchNormV3:batch_mean:02^feature_map_norm4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
"feature_map_norm4/AssignNewValue_1AssignVariableOp<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm4/FusedBatchNormV3:batch_variance:04^feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentity&feature_map_norm4/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????		??
NoOpNoOp)^feature_map_conv1/BiasAdd/ReadVariableOp(^feature_map_conv1/Conv2D/ReadVariableOp)^feature_map_conv2/BiasAdd/ReadVariableOp(^feature_map_conv2/Conv2D/ReadVariableOp)^feature_map_conv3/BiasAdd/ReadVariableOp(^feature_map_conv3/Conv2D/ReadVariableOp)^feature_map_conv4/BiasAdd/ReadVariableOp(^feature_map_conv4/Conv2D/ReadVariableOp!^feature_map_norm1/AssignNewValue#^feature_map_norm1/AssignNewValue_12^feature_map_norm1/FusedBatchNormV3/ReadVariableOp4^feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm1/ReadVariableOp#^feature_map_norm1/ReadVariableOp_1!^feature_map_norm2/AssignNewValue#^feature_map_norm2/AssignNewValue_12^feature_map_norm2/FusedBatchNormV3/ReadVariableOp4^feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm2/ReadVariableOp#^feature_map_norm2/ReadVariableOp_1!^feature_map_norm3/AssignNewValue#^feature_map_norm3/AssignNewValue_12^feature_map_norm3/FusedBatchNormV3/ReadVariableOp4^feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm3/ReadVariableOp#^feature_map_norm3/ReadVariableOp_1!^feature_map_norm4/AssignNewValue#^feature_map_norm4/AssignNewValue_12^feature_map_norm4/FusedBatchNormV3/ReadVariableOp4^feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm4/ReadVariableOp#^feature_map_norm4/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????: : : : : : : : : : : : : : : : : : : : : : : : 2T
(feature_map_conv1/BiasAdd/ReadVariableOp(feature_map_conv1/BiasAdd/ReadVariableOp2R
'feature_map_conv1/Conv2D/ReadVariableOp'feature_map_conv1/Conv2D/ReadVariableOp2T
(feature_map_conv2/BiasAdd/ReadVariableOp(feature_map_conv2/BiasAdd/ReadVariableOp2R
'feature_map_conv2/Conv2D/ReadVariableOp'feature_map_conv2/Conv2D/ReadVariableOp2T
(feature_map_conv3/BiasAdd/ReadVariableOp(feature_map_conv3/BiasAdd/ReadVariableOp2R
'feature_map_conv3/Conv2D/ReadVariableOp'feature_map_conv3/Conv2D/ReadVariableOp2T
(feature_map_conv4/BiasAdd/ReadVariableOp(feature_map_conv4/BiasAdd/ReadVariableOp2R
'feature_map_conv4/Conv2D/ReadVariableOp'feature_map_conv4/Conv2D/ReadVariableOp2D
 feature_map_norm1/AssignNewValue feature_map_norm1/AssignNewValue2H
"feature_map_norm1/AssignNewValue_1"feature_map_norm1/AssignNewValue_12f
1feature_map_norm1/FusedBatchNormV3/ReadVariableOp1feature_map_norm1/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_13feature_map_norm1/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm1/ReadVariableOp feature_map_norm1/ReadVariableOp2H
"feature_map_norm1/ReadVariableOp_1"feature_map_norm1/ReadVariableOp_12D
 feature_map_norm2/AssignNewValue feature_map_norm2/AssignNewValue2H
"feature_map_norm2/AssignNewValue_1"feature_map_norm2/AssignNewValue_12f
1feature_map_norm2/FusedBatchNormV3/ReadVariableOp1feature_map_norm2/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_13feature_map_norm2/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm2/ReadVariableOp feature_map_norm2/ReadVariableOp2H
"feature_map_norm2/ReadVariableOp_1"feature_map_norm2/ReadVariableOp_12D
 feature_map_norm3/AssignNewValue feature_map_norm3/AssignNewValue2H
"feature_map_norm3/AssignNewValue_1"feature_map_norm3/AssignNewValue_12f
1feature_map_norm3/FusedBatchNormV3/ReadVariableOp1feature_map_norm3/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_13feature_map_norm3/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm3/ReadVariableOp feature_map_norm3/ReadVariableOp2H
"feature_map_norm3/ReadVariableOp_1"feature_map_norm3/ReadVariableOp_12D
 feature_map_norm4/AssignNewValue feature_map_norm4/AssignNewValue2H
"feature_map_norm4/AssignNewValue_1"feature_map_norm4/AssignNewValue_12f
1feature_map_norm4/FusedBatchNormV3/ReadVariableOp1feature_map_norm4/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_13feature_map_norm4/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm4/ReadVariableOp feature_map_norm4/ReadVariableOp2H
"feature_map_norm4/ReadVariableOp_1"feature_map_norm4/ReadVariableOp_1:] Y
/
_output_shapes
:?????????
&
_user_specified_nameinput_images
?
?
-__inference_primary_caps_layer_call_fn_726640
feature_maps"
unknown:		?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfeature_mapsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_primary_caps_layer_call_and_return_conditional_losses_724824s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????		?: : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:?????????		?
&
_user_specified_namefeature_maps
?
c
G__inference_digit_probs_layer_call_and_return_conditional_losses_725055

inputs
identityU
norm/mulMulinputsinputs*
T0*+
_output_shapes
:?????????
m
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????
*
	keep_dims(Z
	norm/SqrtSqrtnorm/Sum:output:0*
T0*+
_output_shapes
:?????????
x
norm/SqueezeSqueezenorm/Sqrt:y:0*
T0*'
_output_shapes
:?????????
*
squeeze_dims

?????????]
IdentityIdentitynorm/Squeeze:output:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?"
?
 digit_caps_map_while_body_726174:
6digit_caps_map_while_digit_caps_map_while_loop_counter5
1digit_caps_map_while_digit_caps_map_strided_slice$
 digit_caps_map_while_placeholder&
"digit_caps_map_while_placeholder_19
5digit_caps_map_while_digit_caps_map_strided_slice_1_0u
qdigit_caps_map_while_tensorarrayv2read_tensorlistgetitem_digit_caps_map_tensorarrayunstack_tensorlistfromtensor_0O
5digit_caps_map_while_matmul_readvariableop_resource_0:
!
digit_caps_map_while_identity#
digit_caps_map_while_identity_1#
digit_caps_map_while_identity_2#
digit_caps_map_while_identity_37
3digit_caps_map_while_digit_caps_map_strided_slice_1s
odigit_caps_map_while_tensorarrayv2read_tensorlistgetitem_digit_caps_map_tensorarrayunstack_tensorlistfromtensorM
3digit_caps_map_while_matmul_readvariableop_resource:
??*digit_caps/map/while/MatMul/ReadVariableOp?
Fdigit_caps/map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            ?
8digit_caps/map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqdigit_caps_map_while_tensorarrayv2read_tensorlistgetitem_digit_caps_map_tensorarrayunstack_tensorlistfromtensor_0 digit_caps_map_while_placeholderOdigit_caps/map/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*&
_output_shapes
:
*
element_dtype0?
*digit_caps/map/while/MatMul/ReadVariableOpReadVariableOp5digit_caps_map_while_matmul_readvariableop_resource_0*&
_output_shapes
:
*
dtype0?
digit_caps/map/while/MatMulBatchMatMulV22digit_caps/map/while/MatMul/ReadVariableOp:value:0?digit_caps/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*&
_output_shapes
:
?
9digit_caps/map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"digit_caps_map_while_placeholder_1 digit_caps_map_while_placeholder$digit_caps/map/while/MatMul:output:0*
_output_shapes
: *
element_dtype0:???\
digit_caps/map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
digit_caps/map/while/addAddV2 digit_caps_map_while_placeholder#digit_caps/map/while/add/y:output:0*
T0*
_output_shapes
: ^
digit_caps/map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
digit_caps/map/while/add_1AddV26digit_caps_map_while_digit_caps_map_while_loop_counter%digit_caps/map/while/add_1/y:output:0*
T0*
_output_shapes
: ?
digit_caps/map/while/IdentityIdentitydigit_caps/map/while/add_1:z:0^digit_caps/map/while/NoOp*
T0*
_output_shapes
: ?
digit_caps/map/while/Identity_1Identity1digit_caps_map_while_digit_caps_map_strided_slice^digit_caps/map/while/NoOp*
T0*
_output_shapes
: ?
digit_caps/map/while/Identity_2Identitydigit_caps/map/while/add:z:0^digit_caps/map/while/NoOp*
T0*
_output_shapes
: ?
digit_caps/map/while/Identity_3IdentityIdigit_caps/map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^digit_caps/map/while/NoOp*
T0*
_output_shapes
: :????
digit_caps/map/while/NoOpNoOp+^digit_caps/map/while/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "l
3digit_caps_map_while_digit_caps_map_strided_slice_15digit_caps_map_while_digit_caps_map_strided_slice_1_0"G
digit_caps_map_while_identity&digit_caps/map/while/Identity:output:0"K
digit_caps_map_while_identity_1(digit_caps/map/while/Identity_1:output:0"K
digit_caps_map_while_identity_2(digit_caps/map/while/Identity_2:output:0"K
digit_caps_map_while_identity_3(digit_caps/map/while/Identity_3:output:0"l
3digit_caps_map_while_matmul_readvariableop_resource5digit_caps_map_while_matmul_readvariableop_resource_0"?
odigit_caps_map_while_tensorarrayv2read_tensorlistgetitem_digit_caps_map_tensorarrayunstack_tensorlistfromtensorqdigit_caps_map_while_tensorarrayv2read_tensorlistgetitem_digit_caps_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : : : 2X
*digit_caps/map/while/MatMul/ReadVariableOp*digit_caps/map/while/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
2__inference_feature_map_norm2_layer_call_fn_726914

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_feature_map_norm2_layer_call_and_return_conditional_losses_724476?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
M__inference_feature_map_norm4_layer_call_and_return_conditional_losses_724635

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_feature_map_norm4_layer_call_and_return_conditional_losses_727087

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
2__inference_feature_map_norm4_layer_call_fn_727051

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_feature_map_norm4_layer_call_and_return_conditional_losses_724635?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
map_while_body_726706$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1#
map_while_map_strided_slice_1_0_
[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0D
*map_while_matmul_readvariableop_resource_0:

map_while_identity
map_while_identity_1
map_while_identity_2
map_while_identity_3!
map_while_map_strided_slice_1]
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensorB
(map_while_matmul_readvariableop_resource:
??map/while/MatMul/ReadVariableOp?
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            ?
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholderDmap/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*&
_output_shapes
:
*
element_dtype0?
map/while/MatMul/ReadVariableOpReadVariableOp*map_while_matmul_readvariableop_resource_0*&
_output_shapes
:
*
dtype0?
map/while/MatMulBatchMatMulV2'map/while/MatMul/ReadVariableOp:value:04map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*&
_output_shapes
:
?
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholdermap/while/MatMul:output:0*
_output_shapes
: *
element_dtype0:???Q
map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
map/while/addAddV2map_while_placeholdermap/while/add/y:output:0*
T0*
_output_shapes
: S
map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
map/while/add_1AddV2 map_while_map_while_loop_countermap/while/add_1/y:output:0*
T0*
_output_shapes
: e
map/while/IdentityIdentitymap/while/add_1:z:0^map/while/NoOp*
T0*
_output_shapes
: o
map/while/Identity_1Identitymap_while_map_strided_slice^map/while/NoOp*
T0*
_output_shapes
: e
map/while/Identity_2Identitymap/while/add:z:0^map/while/NoOp*
T0*
_output_shapes
: ?
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^map/while/NoOp*
T0*
_output_shapes
: :???r
map/while/NoOpNoOp ^map/while/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"V
(map_while_matmul_readvariableop_resource*map_while_matmul_readvariableop_resource_0"?
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : : : 2B
map/while/MatMul/ReadVariableOpmap/while/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_digit_caps_layer_call_fn_726684
primary_caps!
unknown:

	unknown_0
	unknown_1:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallprimary_capsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_digit_caps_layer_call_and_return_conditional_losses_724957s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_nameprimary_caps:

_output_shapes
: 
?
?
map_while_cond_724851$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1$
 map_while_less_map_strided_slice<
8map_while_map_while_cond_724851___redundant_placeholder0<
8map_while_map_while_cond_724851___redundant_placeholder1
map_while_identity
p
map/while/LessLessmap_while_placeholder map_while_less_map_strided_slice*
T0*
_output_shapes
: x
map/while/Less_1Less map_while_map_while_loop_countermap_while_map_strided_slice*
T0*
_output_shapes
: d
map/while/LogicalAnd
LogicalAndmap/while/Less_1:z:0map/while/Less:z:0*
_output_shapes
: Y
map/while/IdentityIdentitymap/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "1
map_while_identitymap/while/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
?
?
M__inference_feature_map_norm4_layer_call_and_return_conditional_losses_724604

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_feature_map_norm2_layer_call_and_return_conditional_losses_724476

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
M__inference_feature_map_norm3_layer_call_and_return_conditional_losses_724571

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?O
?
F__inference_digit_caps_layer_call_and_return_conditional_losses_724957
primary_caps+
map_while_input_6:
	
mul_x3
add_3_readvariableop_resource:

identity??add_3/ReadVariableOp?	map/whileP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :y

ExpandDims
ExpandDimsprimary_capsExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????g
Tile/multiplesConst*
_output_shapes
:*
dtype0*%
valueB"   
         t
TileTileExpandDims:output:0Tile/multiples:output:0*
T0*/
_output_shapes
:?????????
_
digit_cap_inputs/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
digit_cap_inputs
ExpandDimsTile:output:0digit_cap_inputs/dim:output:0*
T0*3
_output_shapes!
:?????????
R
	map/ShapeShapedigit_cap_inputs:output:0*
T0*
_output_shapes
:a
map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
map/strided_sliceStridedSlicemap/Shape:output:0 map/strided_slice/stack:output:0"map/strided_slice/stack_1:output:0"map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            ?
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensordigit_cap_inputs:output:0Bmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???K
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : l
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???X
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
	map/whileWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_1:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0map_while_input_6*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : *#
_read_only_resource_inputs
*
_stateful_parallelism( *!
bodyR
map_while_body_724852*!
condR
map_while_cond_724851*!
output_shapes
: : : : : : : ?
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            ?
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:?????????
*
element_dtype0?
digit_cap_predictionsSqueeze/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*/
_output_shapes
:?????????
*
squeeze_dims

??????????
digit_cap_attentionsBatchMatMulV2digit_cap_predictions:output:0digit_cap_predictions:output:0*
T0*/
_output_shapes
:?????????
*
adj_y(j
mulMulmul_xdigit_cap_attentions:output:0*
T0*/
_output_shapes
:?????????
`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????~
SumSummul:z:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????
*
	keep_dims(F
RankConst*
_output_shapes
: *
dtype0*
value	B :P
add/xConst*
_output_shapes
: *
dtype0*
valueB :
?????????L
addAddV2add/x:output:0Rank:output:0*
T0*
_output_shapes
: H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :G
mod/yConst*
_output_shapes
: *
dtype0*
value	B :I
modFloorModadd:z:0mod/y:output:0*
T0*
_output_shapes
: G
Sub/yConst*
_output_shapes
: *
dtype0*
value	B :L
SubSubRank_1:output:0Sub/y:output:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :_
rangeRangerange/start:output:0mod:z:0range/delta:output:0*
_output_shapes
:I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :J
add_1AddV2mod:z:0add_1/y:output:0*
T0*
_output_shapes
: O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :V
range_1Range	add_1:z:0Sub:z:0range_1/delta:output:0*
_output_shapes
: N
concat/values_1PackSub:z:0*
N*
T0*
_output_shapes
:N
concat/values_3Packmod:z:0*
N*
T0*
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2range:output:0concat/values_1:output:0range_1:output:0concat/values_3:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:o
	transpose	TransposeSum:output:0concat:output:0*
T0*/
_output_shapes
:?????????
[
SoftmaxSoftmaxtranspose:y:0*
T0*/
_output_shapes
:?????????
I
Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :P
Sub_1SubRank_1:output:0Sub_1/y:output:0*
T0*
_output_shapes
: O
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
range_2Rangerange_2/start:output:0mod:z:0range_2/delta:output:0*
_output_shapes
:I
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :J
add_2AddV2mod:z:0add_2/y:output:0*
T0*
_output_shapes
: O
range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :X
range_3Range	add_2:z:0	Sub_1:z:0range_3/delta:output:0*
_output_shapes
: R
concat_1/values_1Pack	Sub_1:z:0*
N*
T0*
_output_shapes
:P
concat_1/values_3Packmod:z:0*
N*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_1ConcatV2range_2:output:0concat_1/values_1:output:0range_3:output:0concat_1/values_3:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
digit_cap_coupling_coefficients	TransposeSoftmax:softmax:0concat_1:output:0*
T0*/
_output_shapes
:?????????
v
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*"
_output_shapes
:
*
dtype0?
add_3AddV2#digit_cap_coupling_coefficients:y:0add_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
|
MatMulBatchMatMulV2	add_3:z:0digit_cap_predictions:output:0*
T0*/
_output_shapes
:?????????
y
SqueezeSqueezeMatMul:output:0*
T0*+
_output_shapes
:?????????
*
squeeze_dims

?????????z
digit_cap_squash/norm/mulMulSqueeze:output:0Squeeze:output:0*
T0*+
_output_shapes
:?????????
~
+digit_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
digit_cap_squash/norm/SumSumdigit_cap_squash/norm/mul:z:04digit_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????
*
	keep_dims(|
digit_cap_squash/norm/SqrtSqrt"digit_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:?????????
q
digit_cap_squash/ExpExpdigit_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:?????????
_
digit_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
digit_cap_squash/truedivRealDiv#digit_cap_squash/truediv/x:output:0digit_cap_squash/Exp:y:0*
T0*+
_output_shapes
:?????????
[
digit_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
digit_cap_squash/subSubdigit_cap_squash/sub/x:output:0digit_cap_squash/truediv:z:0*
T0*+
_output_shapes
:?????????
[
digit_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
digit_cap_squash/addAddV2digit_cap_squash/norm/Sqrt:y:0digit_cap_squash/add/y:output:0*
T0*+
_output_shapes
:?????????
?
digit_cap_squash/truediv_1RealDivSqueeze:output:0digit_cap_squash/add:z:0*
T0*+
_output_shapes
:?????????
?
digit_cap_squash/mulMuldigit_cap_squash/sub:z:0digit_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:?????????
k
IdentityIdentitydigit_cap_squash/mul:z:0^NoOp*
T0*+
_output_shapes
:?????????
i
NoOpNoOp^add_3/ReadVariableOp
^map/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2,
add_3/ReadVariableOpadd_3/ReadVariableOp2
	map/while	map/while:Y U
+
_output_shapes
:?????????
&
_user_specified_nameprimary_caps:

_output_shapes
: 
?
H
,__inference_digit_probs_layer_call_fn_726816

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_digit_probs_layer_call_and_return_conditional_losses_724974`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
2__inference_feature_map_norm2_layer_call_fn_726927

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_feature_map_norm2_layer_call_and_return_conditional_losses_724507?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
2__inference_Efficient-CapsNet_layer_call_fn_725038
input_images!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?%

unknown_23:		?

unknown_24:	?$

unknown_25:


unknown_26 

unknown_27:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_imagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_724977o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameinput_images:

_output_shapes
: 
?'
?
H__inference_primary_caps_layer_call_and_return_conditional_losses_726673
feature_mapsK
0primary_cap_dconv_conv2d_readvariableop_resource:		?@
1primary_cap_dconv_biasadd_readvariableop_resource:	?
identity??(primary_cap_dconv/BiasAdd/ReadVariableOp?'primary_cap_dconv/Conv2D/ReadVariableOp?
'primary_cap_dconv/Conv2D/ReadVariableOpReadVariableOp0primary_cap_dconv_conv2d_readvariableop_resource*'
_output_shapes
:		?*
dtype0?
primary_cap_dconv/Conv2DConv2Dfeature_maps/primary_cap_dconv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
(primary_cap_dconv/BiasAdd/ReadVariableOpReadVariableOp1primary_cap_dconv_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
primary_cap_dconv/BiasAddBiasAdd!primary_cap_dconv/Conv2D:output:00primary_cap_dconv/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????}
primary_cap_dconv/ReluRelu"primary_cap_dconv/BiasAdd:output:0*
T0*0
_output_shapes
:??????????m
primary_cap_reshape/ShapeShape$primary_cap_dconv/Relu:activations:0*
T0*
_output_shapes
:q
'primary_cap_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)primary_cap_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)primary_cap_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!primary_cap_reshape/strided_sliceStridedSlice"primary_cap_reshape/Shape:output:00primary_cap_reshape/strided_slice/stack:output:02primary_cap_reshape/strided_slice/stack_1:output:02primary_cap_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#primary_cap_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????e
#primary_cap_reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
!primary_cap_reshape/Reshape/shapePack*primary_cap_reshape/strided_slice:output:0,primary_cap_reshape/Reshape/shape/1:output:0,primary_cap_reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
primary_cap_reshape/ReshapeReshape$primary_cap_dconv/Relu:activations:0*primary_cap_reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:??????????
primary_cap_squash/norm/mulMul$primary_cap_reshape/Reshape:output:0$primary_cap_reshape/Reshape:output:0*
T0*+
_output_shapes
:??????????
-primary_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
primary_cap_squash/norm/SumSumprimary_cap_squash/norm/mul:z:06primary_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(?
primary_cap_squash/norm/SqrtSqrt$primary_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:?????????u
primary_cap_squash/ExpExp primary_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:?????????a
primary_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
primary_cap_squash/truedivRealDiv%primary_cap_squash/truediv/x:output:0primary_cap_squash/Exp:y:0*
T0*+
_output_shapes
:?????????]
primary_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
primary_cap_squash/subSub!primary_cap_squash/sub/x:output:0primary_cap_squash/truediv:z:0*
T0*+
_output_shapes
:?????????]
primary_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
primary_cap_squash/addAddV2 primary_cap_squash/norm/Sqrt:y:0!primary_cap_squash/add/y:output:0*
T0*+
_output_shapes
:??????????
primary_cap_squash/truediv_1RealDiv$primary_cap_reshape/Reshape:output:0primary_cap_squash/add:z:0*
T0*+
_output_shapes
:??????????
primary_cap_squash/mulMulprimary_cap_squash/sub:z:0 primary_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:?????????m
IdentityIdentityprimary_cap_squash/mul:z:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp)^primary_cap_dconv/BiasAdd/ReadVariableOp(^primary_cap_dconv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????		?: : 2T
(primary_cap_dconv/BiasAdd/ReadVariableOp(primary_cap_dconv/BiasAdd/ReadVariableOp2R
'primary_cap_dconv/Conv2D/ReadVariableOp'primary_cap_dconv/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:?????????		?
&
_user_specified_namefeature_maps
?
H
,__inference_digit_probs_layer_call_fn_726821

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_digit_probs_layer_call_and_return_conditional_losses_725055`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
map_while_cond_726705$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1$
 map_while_less_map_strided_slice<
8map_while_map_while_cond_726705___redundant_placeholder0<
8map_while_map_while_cond_726705___redundant_placeholder1
map_while_identity
p
map/while/LessLessmap_while_placeholder map_while_less_map_strided_slice*
T0*
_output_shapes
: x
map/while/Less_1Less map_while_map_while_loop_countermap_while_map_strided_slice*
T0*
_output_shapes
: d
map/while/LogicalAnd
LogicalAndmap/while/Less_1:z:0map/while/Less:z:0*
_output_shapes
: Y
map/while/IdentityIdentitymap/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "1
map_while_identitymap/while/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
?!
?

M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_725406

inputs-
feature_maps_725343: !
feature_maps_725345: !
feature_maps_725347: !
feature_maps_725349: !
feature_maps_725351: !
feature_maps_725353: -
feature_maps_725355: @!
feature_maps_725357:@!
feature_maps_725359:@!
feature_maps_725361:@!
feature_maps_725363:@!
feature_maps_725365:@-
feature_maps_725367:@@!
feature_maps_725369:@!
feature_maps_725371:@!
feature_maps_725373:@!
feature_maps_725375:@!
feature_maps_725377:@.
feature_maps_725379:@?"
feature_maps_725381:	?"
feature_maps_725383:	?"
feature_maps_725385:	?"
feature_maps_725387:	?"
feature_maps_725389:	?.
primary_caps_725392:		?"
primary_caps_725394:	?+
digit_caps_725397:

digit_caps_725399'
digit_caps_725401:

identity??"digit_caps/StatefulPartitionedCall?$feature_maps/StatefulPartitionedCall?$primary_caps/StatefulPartitionedCall?
$feature_maps/StatefulPartitionedCallStatefulPartitionedCallinputsfeature_maps_725343feature_maps_725345feature_maps_725347feature_maps_725349feature_maps_725351feature_maps_725353feature_maps_725355feature_maps_725357feature_maps_725359feature_maps_725361feature_maps_725363feature_maps_725365feature_maps_725367feature_maps_725369feature_maps_725371feature_maps_725373feature_maps_725375feature_maps_725377feature_maps_725379feature_maps_725381feature_maps_725383feature_maps_725385feature_maps_725387feature_maps_725389*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_feature_maps_layer_call_and_return_conditional_losses_725224?
$primary_caps/StatefulPartitionedCallStatefulPartitionedCall-feature_maps/StatefulPartitionedCall:output:0primary_caps_725392primary_caps_725394*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_primary_caps_layer_call_and_return_conditional_losses_724824?
"digit_caps/StatefulPartitionedCallStatefulPartitionedCall-primary_caps/StatefulPartitionedCall:output:0digit_caps_725397digit_caps_725399digit_caps_725401*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_digit_caps_layer_call_and_return_conditional_losses_724957?
digit_probs/PartitionedCallPartitionedCall+digit_caps/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_digit_probs_layer_call_and_return_conditional_losses_725055s
IdentityIdentity$digit_probs/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp#^digit_caps/StatefulPartitionedCall%^feature_maps/StatefulPartitionedCall%^primary_caps/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"digit_caps/StatefulPartitionedCall"digit_caps/StatefulPartitionedCall2L
$feature_maps/StatefulPartitionedCall$feature_maps/StatefulPartitionedCall2L
$primary_caps/StatefulPartitionedCall$primary_caps/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?m
?
H__inference_feature_maps_layer_call_and_return_conditional_losses_724741
input_imagesJ
0feature_map_conv1_conv2d_readvariableop_resource: ?
1feature_map_conv1_biasadd_readvariableop_resource: 7
)feature_map_norm1_readvariableop_resource: 9
+feature_map_norm1_readvariableop_1_resource: H
:feature_map_norm1_fusedbatchnormv3_readvariableop_resource: J
<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource: J
0feature_map_conv2_conv2d_readvariableop_resource: @?
1feature_map_conv2_biasadd_readvariableop_resource:@7
)feature_map_norm2_readvariableop_resource:@9
+feature_map_norm2_readvariableop_1_resource:@H
:feature_map_norm2_fusedbatchnormv3_readvariableop_resource:@J
<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource:@J
0feature_map_conv3_conv2d_readvariableop_resource:@@?
1feature_map_conv3_biasadd_readvariableop_resource:@7
)feature_map_norm3_readvariableop_resource:@9
+feature_map_norm3_readvariableop_1_resource:@H
:feature_map_norm3_fusedbatchnormv3_readvariableop_resource:@J
<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource:@K
0feature_map_conv4_conv2d_readvariableop_resource:@?@
1feature_map_conv4_biasadd_readvariableop_resource:	?8
)feature_map_norm4_readvariableop_resource:	?:
+feature_map_norm4_readvariableop_1_resource:	?I
:feature_map_norm4_fusedbatchnormv3_readvariableop_resource:	?K
<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource:	?
identity??(feature_map_conv1/BiasAdd/ReadVariableOp?'feature_map_conv1/Conv2D/ReadVariableOp?(feature_map_conv2/BiasAdd/ReadVariableOp?'feature_map_conv2/Conv2D/ReadVariableOp?(feature_map_conv3/BiasAdd/ReadVariableOp?'feature_map_conv3/Conv2D/ReadVariableOp?(feature_map_conv4/BiasAdd/ReadVariableOp?'feature_map_conv4/Conv2D/ReadVariableOp?1feature_map_norm1/FusedBatchNormV3/ReadVariableOp?3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1? feature_map_norm1/ReadVariableOp?"feature_map_norm1/ReadVariableOp_1?1feature_map_norm2/FusedBatchNormV3/ReadVariableOp?3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1? feature_map_norm2/ReadVariableOp?"feature_map_norm2/ReadVariableOp_1?1feature_map_norm3/FusedBatchNormV3/ReadVariableOp?3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1? feature_map_norm3/ReadVariableOp?"feature_map_norm3/ReadVariableOp_1?1feature_map_norm4/FusedBatchNormV3/ReadVariableOp?3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1? feature_map_norm4/ReadVariableOp?"feature_map_norm4/ReadVariableOp_1?
'feature_map_conv1/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
feature_map_conv1/Conv2DConv2Dinput_images/feature_map_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
(feature_map_conv1/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
feature_map_conv1/BiasAddBiasAdd!feature_map_conv1/Conv2D:output:00feature_map_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? |
feature_map_conv1/ReluRelu"feature_map_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
 feature_map_norm1/ReadVariableOpReadVariableOp)feature_map_norm1_readvariableop_resource*
_output_shapes
: *
dtype0?
"feature_map_norm1/ReadVariableOp_1ReadVariableOp+feature_map_norm1_readvariableop_1_resource*
_output_shapes
: *
dtype0?
1feature_map_norm1/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
"feature_map_norm1/FusedBatchNormV3FusedBatchNormV3$feature_map_conv1/Relu:activations:0(feature_map_norm1/ReadVariableOp:value:0*feature_map_norm1/ReadVariableOp_1:value:09feature_map_norm1/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
'feature_map_conv2/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
feature_map_conv2/Conv2DConv2D&feature_map_norm1/FusedBatchNormV3:y:0/feature_map_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
(feature_map_conv2/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
feature_map_conv2/BiasAddBiasAdd!feature_map_conv2/Conv2D:output:00feature_map_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@|
feature_map_conv2/ReluRelu"feature_map_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
 feature_map_norm2/ReadVariableOpReadVariableOp)feature_map_norm2_readvariableop_resource*
_output_shapes
:@*
dtype0?
"feature_map_norm2/ReadVariableOp_1ReadVariableOp+feature_map_norm2_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
1feature_map_norm2/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
"feature_map_norm2/FusedBatchNormV3FusedBatchNormV3$feature_map_conv2/Relu:activations:0(feature_map_norm2/ReadVariableOp:value:0*feature_map_norm2/ReadVariableOp_1:value:09feature_map_norm2/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
'feature_map_conv3/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
feature_map_conv3/Conv2DConv2D&feature_map_norm2/FusedBatchNormV3:y:0/feature_map_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
(feature_map_conv3/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
feature_map_conv3/BiasAddBiasAdd!feature_map_conv3/Conv2D:output:00feature_map_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@|
feature_map_conv3/ReluRelu"feature_map_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
 feature_map_norm3/ReadVariableOpReadVariableOp)feature_map_norm3_readvariableop_resource*
_output_shapes
:@*
dtype0?
"feature_map_norm3/ReadVariableOp_1ReadVariableOp+feature_map_norm3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
1feature_map_norm3/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
"feature_map_norm3/FusedBatchNormV3FusedBatchNormV3$feature_map_conv3/Relu:activations:0(feature_map_norm3/ReadVariableOp:value:0*feature_map_norm3/ReadVariableOp_1:value:09feature_map_norm3/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
'feature_map_conv4/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
feature_map_conv4/Conv2DConv2D&feature_map_norm3/FusedBatchNormV3:y:0/feature_map_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
?
(feature_map_conv4/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
feature_map_conv4/BiasAddBiasAdd!feature_map_conv4/Conv2D:output:00feature_map_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?}
feature_map_conv4/ReluRelu"feature_map_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		??
 feature_map_norm4/ReadVariableOpReadVariableOp)feature_map_norm4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"feature_map_norm4/ReadVariableOp_1ReadVariableOp+feature_map_norm4_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
1feature_map_norm4/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
"feature_map_norm4/FusedBatchNormV3FusedBatchNormV3$feature_map_conv4/Relu:activations:0(feature_map_norm4/ReadVariableOp:value:0*feature_map_norm4/ReadVariableOp_1:value:09feature_map_norm4/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????		?:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentity&feature_map_norm4/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????		??
NoOpNoOp)^feature_map_conv1/BiasAdd/ReadVariableOp(^feature_map_conv1/Conv2D/ReadVariableOp)^feature_map_conv2/BiasAdd/ReadVariableOp(^feature_map_conv2/Conv2D/ReadVariableOp)^feature_map_conv3/BiasAdd/ReadVariableOp(^feature_map_conv3/Conv2D/ReadVariableOp)^feature_map_conv4/BiasAdd/ReadVariableOp(^feature_map_conv4/Conv2D/ReadVariableOp2^feature_map_norm1/FusedBatchNormV3/ReadVariableOp4^feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm1/ReadVariableOp#^feature_map_norm1/ReadVariableOp_12^feature_map_norm2/FusedBatchNormV3/ReadVariableOp4^feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm2/ReadVariableOp#^feature_map_norm2/ReadVariableOp_12^feature_map_norm3/FusedBatchNormV3/ReadVariableOp4^feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm3/ReadVariableOp#^feature_map_norm3/ReadVariableOp_12^feature_map_norm4/FusedBatchNormV3/ReadVariableOp4^feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm4/ReadVariableOp#^feature_map_norm4/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????: : : : : : : : : : : : : : : : : : : : : : : : 2T
(feature_map_conv1/BiasAdd/ReadVariableOp(feature_map_conv1/BiasAdd/ReadVariableOp2R
'feature_map_conv1/Conv2D/ReadVariableOp'feature_map_conv1/Conv2D/ReadVariableOp2T
(feature_map_conv2/BiasAdd/ReadVariableOp(feature_map_conv2/BiasAdd/ReadVariableOp2R
'feature_map_conv2/Conv2D/ReadVariableOp'feature_map_conv2/Conv2D/ReadVariableOp2T
(feature_map_conv3/BiasAdd/ReadVariableOp(feature_map_conv3/BiasAdd/ReadVariableOp2R
'feature_map_conv3/Conv2D/ReadVariableOp'feature_map_conv3/Conv2D/ReadVariableOp2T
(feature_map_conv4/BiasAdd/ReadVariableOp(feature_map_conv4/BiasAdd/ReadVariableOp2R
'feature_map_conv4/Conv2D/ReadVariableOp'feature_map_conv4/Conv2D/ReadVariableOp2f
1feature_map_norm1/FusedBatchNormV3/ReadVariableOp1feature_map_norm1/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_13feature_map_norm1/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm1/ReadVariableOp feature_map_norm1/ReadVariableOp2H
"feature_map_norm1/ReadVariableOp_1"feature_map_norm1/ReadVariableOp_12f
1feature_map_norm2/FusedBatchNormV3/ReadVariableOp1feature_map_norm2/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_13feature_map_norm2/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm2/ReadVariableOp feature_map_norm2/ReadVariableOp2H
"feature_map_norm2/ReadVariableOp_1"feature_map_norm2/ReadVariableOp_12f
1feature_map_norm3/FusedBatchNormV3/ReadVariableOp1feature_map_norm3/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_13feature_map_norm3/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm3/ReadVariableOp feature_map_norm3/ReadVariableOp2H
"feature_map_norm3/ReadVariableOp_1"feature_map_norm3/ReadVariableOp_12f
1feature_map_norm4/FusedBatchNormV3/ReadVariableOp1feature_map_norm4/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_13feature_map_norm4/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm4/ReadVariableOp feature_map_norm4/ReadVariableOp2H
"feature_map_norm4/ReadVariableOp_1"feature_map_norm4/ReadVariableOp_1:] Y
/
_output_shapes
:?????????
&
_user_specified_nameinput_images
?
?
M__inference_feature_map_norm1_layer_call_and_return_conditional_losses_726901

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
2__inference_feature_map_norm1_layer_call_fn_726852

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_feature_map_norm1_layer_call_and_return_conditional_losses_724412?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?+
__inference__traced_save_727342
file_prefixE
Asavev2_digit_caps_digit_caps_transform_tensor_read_readvariableop?
;savev2_digit_caps_digit_caps_log_priors_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopD
@savev2_feature_maps_feature_map_conv1_kernel_read_readvariableopB
>savev2_feature_maps_feature_map_conv1_bias_read_readvariableopC
?savev2_feature_maps_feature_map_norm1_gamma_read_readvariableopB
>savev2_feature_maps_feature_map_norm1_beta_read_readvariableopD
@savev2_feature_maps_feature_map_conv2_kernel_read_readvariableopB
>savev2_feature_maps_feature_map_conv2_bias_read_readvariableopC
?savev2_feature_maps_feature_map_norm2_gamma_read_readvariableopB
>savev2_feature_maps_feature_map_norm2_beta_read_readvariableopD
@savev2_feature_maps_feature_map_conv3_kernel_read_readvariableopB
>savev2_feature_maps_feature_map_conv3_bias_read_readvariableopC
?savev2_feature_maps_feature_map_norm3_gamma_read_readvariableopB
>savev2_feature_maps_feature_map_norm3_beta_read_readvariableopD
@savev2_feature_maps_feature_map_conv4_kernel_read_readvariableopB
>savev2_feature_maps_feature_map_conv4_bias_read_readvariableopC
?savev2_feature_maps_feature_map_norm4_gamma_read_readvariableopB
>savev2_feature_maps_feature_map_norm4_beta_read_readvariableopI
Esavev2_feature_maps_feature_map_norm1_moving_mean_read_readvariableopM
Isavev2_feature_maps_feature_map_norm1_moving_variance_read_readvariableopI
Esavev2_feature_maps_feature_map_norm2_moving_mean_read_readvariableopM
Isavev2_feature_maps_feature_map_norm2_moving_variance_read_readvariableopI
Esavev2_feature_maps_feature_map_norm3_moving_mean_read_readvariableopM
Isavev2_feature_maps_feature_map_norm3_moving_variance_read_readvariableopI
Esavev2_feature_maps_feature_map_norm4_moving_mean_read_readvariableopM
Isavev2_feature_maps_feature_map_norm4_moving_variance_read_readvariableopD
@savev2_primary_caps_primary_cap_dconv_kernel_read_readvariableopB
>savev2_primary_caps_primary_cap_dconv_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopL
Hsavev2_adam_digit_caps_digit_caps_transform_tensor_m_read_readvariableopF
Bsavev2_adam_digit_caps_digit_caps_log_priors_m_read_readvariableopK
Gsavev2_adam_feature_maps_feature_map_conv1_kernel_m_read_readvariableopI
Esavev2_adam_feature_maps_feature_map_conv1_bias_m_read_readvariableopJ
Fsavev2_adam_feature_maps_feature_map_norm1_gamma_m_read_readvariableopI
Esavev2_adam_feature_maps_feature_map_norm1_beta_m_read_readvariableopK
Gsavev2_adam_feature_maps_feature_map_conv2_kernel_m_read_readvariableopI
Esavev2_adam_feature_maps_feature_map_conv2_bias_m_read_readvariableopJ
Fsavev2_adam_feature_maps_feature_map_norm2_gamma_m_read_readvariableopI
Esavev2_adam_feature_maps_feature_map_norm2_beta_m_read_readvariableopK
Gsavev2_adam_feature_maps_feature_map_conv3_kernel_m_read_readvariableopI
Esavev2_adam_feature_maps_feature_map_conv3_bias_m_read_readvariableopJ
Fsavev2_adam_feature_maps_feature_map_norm3_gamma_m_read_readvariableopI
Esavev2_adam_feature_maps_feature_map_norm3_beta_m_read_readvariableopK
Gsavev2_adam_feature_maps_feature_map_conv4_kernel_m_read_readvariableopI
Esavev2_adam_feature_maps_feature_map_conv4_bias_m_read_readvariableopJ
Fsavev2_adam_feature_maps_feature_map_norm4_gamma_m_read_readvariableopI
Esavev2_adam_feature_maps_feature_map_norm4_beta_m_read_readvariableopK
Gsavev2_adam_primary_caps_primary_cap_dconv_kernel_m_read_readvariableopI
Esavev2_adam_primary_caps_primary_cap_dconv_bias_m_read_readvariableopL
Hsavev2_adam_digit_caps_digit_caps_transform_tensor_v_read_readvariableopF
Bsavev2_adam_digit_caps_digit_caps_log_priors_v_read_readvariableopK
Gsavev2_adam_feature_maps_feature_map_conv1_kernel_v_read_readvariableopI
Esavev2_adam_feature_maps_feature_map_conv1_bias_v_read_readvariableopJ
Fsavev2_adam_feature_maps_feature_map_norm1_gamma_v_read_readvariableopI
Esavev2_adam_feature_maps_feature_map_norm1_beta_v_read_readvariableopK
Gsavev2_adam_feature_maps_feature_map_conv2_kernel_v_read_readvariableopI
Esavev2_adam_feature_maps_feature_map_conv2_bias_v_read_readvariableopJ
Fsavev2_adam_feature_maps_feature_map_norm2_gamma_v_read_readvariableopI
Esavev2_adam_feature_maps_feature_map_norm2_beta_v_read_readvariableopK
Gsavev2_adam_feature_maps_feature_map_conv3_kernel_v_read_readvariableopI
Esavev2_adam_feature_maps_feature_map_conv3_bias_v_read_readvariableopJ
Fsavev2_adam_feature_maps_feature_map_norm3_gamma_v_read_readvariableopI
Esavev2_adam_feature_maps_feature_map_norm3_beta_v_read_readvariableopK
Gsavev2_adam_feature_maps_feature_map_conv4_kernel_v_read_readvariableopI
Esavev2_adam_feature_maps_feature_map_conv4_bias_v_read_readvariableopJ
Fsavev2_adam_feature_maps_feature_map_norm4_gamma_v_read_readvariableopI
Esavev2_adam_feature_maps_feature_map_norm4_beta_v_read_readvariableopK
Gsavev2_adam_primary_caps_primary_cap_dconv_kernel_v_read_readvariableopI
Esavev2_adam_primary_caps_primary_cap_dconv_bias_v_read_readvariableop
savev2_const_1

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?$
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*?#
value?#B?#NBKlayer_with_weights-2/digit_caps_transform_tensor/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-2/digit_caps_log_priors/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBglayer_with_weights-2/digit_caps_transform_tensor/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBalayer_with_weights-2/digit_caps_log_priors/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBglayer_with_weights-2/digit_caps_transform_tensor/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBalayer_with_weights-2/digit_caps_log_priors/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*?
value?B?NB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?)
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Asavev2_digit_caps_digit_caps_transform_tensor_read_readvariableop;savev2_digit_caps_digit_caps_log_priors_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop@savev2_feature_maps_feature_map_conv1_kernel_read_readvariableop>savev2_feature_maps_feature_map_conv1_bias_read_readvariableop?savev2_feature_maps_feature_map_norm1_gamma_read_readvariableop>savev2_feature_maps_feature_map_norm1_beta_read_readvariableop@savev2_feature_maps_feature_map_conv2_kernel_read_readvariableop>savev2_feature_maps_feature_map_conv2_bias_read_readvariableop?savev2_feature_maps_feature_map_norm2_gamma_read_readvariableop>savev2_feature_maps_feature_map_norm2_beta_read_readvariableop@savev2_feature_maps_feature_map_conv3_kernel_read_readvariableop>savev2_feature_maps_feature_map_conv3_bias_read_readvariableop?savev2_feature_maps_feature_map_norm3_gamma_read_readvariableop>savev2_feature_maps_feature_map_norm3_beta_read_readvariableop@savev2_feature_maps_feature_map_conv4_kernel_read_readvariableop>savev2_feature_maps_feature_map_conv4_bias_read_readvariableop?savev2_feature_maps_feature_map_norm4_gamma_read_readvariableop>savev2_feature_maps_feature_map_norm4_beta_read_readvariableopEsavev2_feature_maps_feature_map_norm1_moving_mean_read_readvariableopIsavev2_feature_maps_feature_map_norm1_moving_variance_read_readvariableopEsavev2_feature_maps_feature_map_norm2_moving_mean_read_readvariableopIsavev2_feature_maps_feature_map_norm2_moving_variance_read_readvariableopEsavev2_feature_maps_feature_map_norm3_moving_mean_read_readvariableopIsavev2_feature_maps_feature_map_norm3_moving_variance_read_readvariableopEsavev2_feature_maps_feature_map_norm4_moving_mean_read_readvariableopIsavev2_feature_maps_feature_map_norm4_moving_variance_read_readvariableop@savev2_primary_caps_primary_cap_dconv_kernel_read_readvariableop>savev2_primary_caps_primary_cap_dconv_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopHsavev2_adam_digit_caps_digit_caps_transform_tensor_m_read_readvariableopBsavev2_adam_digit_caps_digit_caps_log_priors_m_read_readvariableopGsavev2_adam_feature_maps_feature_map_conv1_kernel_m_read_readvariableopEsavev2_adam_feature_maps_feature_map_conv1_bias_m_read_readvariableopFsavev2_adam_feature_maps_feature_map_norm1_gamma_m_read_readvariableopEsavev2_adam_feature_maps_feature_map_norm1_beta_m_read_readvariableopGsavev2_adam_feature_maps_feature_map_conv2_kernel_m_read_readvariableopEsavev2_adam_feature_maps_feature_map_conv2_bias_m_read_readvariableopFsavev2_adam_feature_maps_feature_map_norm2_gamma_m_read_readvariableopEsavev2_adam_feature_maps_feature_map_norm2_beta_m_read_readvariableopGsavev2_adam_feature_maps_feature_map_conv3_kernel_m_read_readvariableopEsavev2_adam_feature_maps_feature_map_conv3_bias_m_read_readvariableopFsavev2_adam_feature_maps_feature_map_norm3_gamma_m_read_readvariableopEsavev2_adam_feature_maps_feature_map_norm3_beta_m_read_readvariableopGsavev2_adam_feature_maps_feature_map_conv4_kernel_m_read_readvariableopEsavev2_adam_feature_maps_feature_map_conv4_bias_m_read_readvariableopFsavev2_adam_feature_maps_feature_map_norm4_gamma_m_read_readvariableopEsavev2_adam_feature_maps_feature_map_norm4_beta_m_read_readvariableopGsavev2_adam_primary_caps_primary_cap_dconv_kernel_m_read_readvariableopEsavev2_adam_primary_caps_primary_cap_dconv_bias_m_read_readvariableopHsavev2_adam_digit_caps_digit_caps_transform_tensor_v_read_readvariableopBsavev2_adam_digit_caps_digit_caps_log_priors_v_read_readvariableopGsavev2_adam_feature_maps_feature_map_conv1_kernel_v_read_readvariableopEsavev2_adam_feature_maps_feature_map_conv1_bias_v_read_readvariableopFsavev2_adam_feature_maps_feature_map_norm1_gamma_v_read_readvariableopEsavev2_adam_feature_maps_feature_map_norm1_beta_v_read_readvariableopGsavev2_adam_feature_maps_feature_map_conv2_kernel_v_read_readvariableopEsavev2_adam_feature_maps_feature_map_conv2_bias_v_read_readvariableopFsavev2_adam_feature_maps_feature_map_norm2_gamma_v_read_readvariableopEsavev2_adam_feature_maps_feature_map_norm2_beta_v_read_readvariableopGsavev2_adam_feature_maps_feature_map_conv3_kernel_v_read_readvariableopEsavev2_adam_feature_maps_feature_map_conv3_bias_v_read_readvariableopFsavev2_adam_feature_maps_feature_map_norm3_gamma_v_read_readvariableopEsavev2_adam_feature_maps_feature_map_norm3_beta_v_read_readvariableopGsavev2_adam_feature_maps_feature_map_conv4_kernel_v_read_readvariableopEsavev2_adam_feature_maps_feature_map_conv4_bias_v_read_readvariableopFsavev2_adam_feature_maps_feature_map_norm4_gamma_v_read_readvariableopEsavev2_adam_feature_maps_feature_map_norm4_beta_v_read_readvariableopGsavev2_adam_primary_caps_primary_cap_dconv_kernel_v_read_readvariableopEsavev2_adam_primary_caps_primary_cap_dconv_bias_v_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *\
dtypesR
P2N	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
:
: : : : : : : : : : @:@:@:@:@@:@:@:@:@?:?:?:?: : :@:@:@:@:?:?:		?:?: : : : :
:
: : : : : @:@:@:@:@@:@:@:@:@?:?:?:?:		?:?:
:
: : : : : @:@:@:@:@@:@:@:@:@?:?:?:?:		?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:
:($
"
_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?: 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:!

_output_shapes	
:?:!

_output_shapes	
:?:- )
'
_output_shapes
:		?:!!

_output_shapes	
:?:"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :,&(
&
_output_shapes
:
:('$
"
_output_shapes
:
:,((
&
_output_shapes
: : )

_output_shapes
: : *

_output_shapes
: : +

_output_shapes
: :,,(
&
_output_shapes
: @: -

_output_shapes
:@: .

_output_shapes
:@: /

_output_shapes
:@:,0(
&
_output_shapes
:@@: 1

_output_shapes
:@: 2

_output_shapes
:@: 3

_output_shapes
:@:-4)
'
_output_shapes
:@?:!5

_output_shapes	
:?:!6

_output_shapes	
:?:!7

_output_shapes	
:?:-8)
'
_output_shapes
:		?:!9

_output_shapes	
:?:,:(
&
_output_shapes
:
:(;$
"
_output_shapes
:
:,<(
&
_output_shapes
: : =

_output_shapes
: : >

_output_shapes
: : ?

_output_shapes
: :,@(
&
_output_shapes
: @: A

_output_shapes
:@: B

_output_shapes
:@: C

_output_shapes
:@:,D(
&
_output_shapes
:@@: E

_output_shapes
:@: F

_output_shapes
:@: G

_output_shapes
:@:-H)
'
_output_shapes
:@?:!I

_output_shapes	
:?:!J

_output_shapes	
:?:!K

_output_shapes	
:?:-L)
'
_output_shapes
:		?:!M

_output_shapes	
:?:N

_output_shapes
: 
?
c
G__inference_digit_probs_layer_call_and_return_conditional_losses_726830

inputs
identityU
norm/mulMulinputsinputs*
T0*+
_output_shapes
:?????????
m
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????
*
	keep_dims(Z
	norm/SqrtSqrtnorm/Sum:output:0*
T0*+
_output_shapes
:?????????
x
norm/SqueezeSqueezenorm/Sqrt:y:0*
T0*'
_output_shapes
:?????????
*
squeeze_dims

?????????]
IdentityIdentitynorm/Squeeze:output:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
M__inference_feature_map_norm2_layer_call_and_return_conditional_losses_724507

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?;
"__inference__traced_restore_727583
file_prefixQ
7assignvariableop_digit_caps_digit_caps_transform_tensor:
I
3assignvariableop_1_digit_caps_digit_caps_log_priors:
&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: R
8assignvariableop_7_feature_maps_feature_map_conv1_kernel: D
6assignvariableop_8_feature_maps_feature_map_conv1_bias: E
7assignvariableop_9_feature_maps_feature_map_norm1_gamma: E
7assignvariableop_10_feature_maps_feature_map_norm1_beta: S
9assignvariableop_11_feature_maps_feature_map_conv2_kernel: @E
7assignvariableop_12_feature_maps_feature_map_conv2_bias:@F
8assignvariableop_13_feature_maps_feature_map_norm2_gamma:@E
7assignvariableop_14_feature_maps_feature_map_norm2_beta:@S
9assignvariableop_15_feature_maps_feature_map_conv3_kernel:@@E
7assignvariableop_16_feature_maps_feature_map_conv3_bias:@F
8assignvariableop_17_feature_maps_feature_map_norm3_gamma:@E
7assignvariableop_18_feature_maps_feature_map_norm3_beta:@T
9assignvariableop_19_feature_maps_feature_map_conv4_kernel:@?F
7assignvariableop_20_feature_maps_feature_map_conv4_bias:	?G
8assignvariableop_21_feature_maps_feature_map_norm4_gamma:	?F
7assignvariableop_22_feature_maps_feature_map_norm4_beta:	?L
>assignvariableop_23_feature_maps_feature_map_norm1_moving_mean: P
Bassignvariableop_24_feature_maps_feature_map_norm1_moving_variance: L
>assignvariableop_25_feature_maps_feature_map_norm2_moving_mean:@P
Bassignvariableop_26_feature_maps_feature_map_norm2_moving_variance:@L
>assignvariableop_27_feature_maps_feature_map_norm3_moving_mean:@P
Bassignvariableop_28_feature_maps_feature_map_norm3_moving_variance:@M
>assignvariableop_29_feature_maps_feature_map_norm4_moving_mean:	?Q
Bassignvariableop_30_feature_maps_feature_map_norm4_moving_variance:	?T
9assignvariableop_31_primary_caps_primary_cap_dconv_kernel:		?F
7assignvariableop_32_primary_caps_primary_cap_dconv_bias:	?#
assignvariableop_33_total: #
assignvariableop_34_count: %
assignvariableop_35_total_1: %
assignvariableop_36_count_1: [
Aassignvariableop_37_adam_digit_caps_digit_caps_transform_tensor_m:
Q
;assignvariableop_38_adam_digit_caps_digit_caps_log_priors_m:
Z
@assignvariableop_39_adam_feature_maps_feature_map_conv1_kernel_m: L
>assignvariableop_40_adam_feature_maps_feature_map_conv1_bias_m: M
?assignvariableop_41_adam_feature_maps_feature_map_norm1_gamma_m: L
>assignvariableop_42_adam_feature_maps_feature_map_norm1_beta_m: Z
@assignvariableop_43_adam_feature_maps_feature_map_conv2_kernel_m: @L
>assignvariableop_44_adam_feature_maps_feature_map_conv2_bias_m:@M
?assignvariableop_45_adam_feature_maps_feature_map_norm2_gamma_m:@L
>assignvariableop_46_adam_feature_maps_feature_map_norm2_beta_m:@Z
@assignvariableop_47_adam_feature_maps_feature_map_conv3_kernel_m:@@L
>assignvariableop_48_adam_feature_maps_feature_map_conv3_bias_m:@M
?assignvariableop_49_adam_feature_maps_feature_map_norm3_gamma_m:@L
>assignvariableop_50_adam_feature_maps_feature_map_norm3_beta_m:@[
@assignvariableop_51_adam_feature_maps_feature_map_conv4_kernel_m:@?M
>assignvariableop_52_adam_feature_maps_feature_map_conv4_bias_m:	?N
?assignvariableop_53_adam_feature_maps_feature_map_norm4_gamma_m:	?M
>assignvariableop_54_adam_feature_maps_feature_map_norm4_beta_m:	?[
@assignvariableop_55_adam_primary_caps_primary_cap_dconv_kernel_m:		?M
>assignvariableop_56_adam_primary_caps_primary_cap_dconv_bias_m:	?[
Aassignvariableop_57_adam_digit_caps_digit_caps_transform_tensor_v:
Q
;assignvariableop_58_adam_digit_caps_digit_caps_log_priors_v:
Z
@assignvariableop_59_adam_feature_maps_feature_map_conv1_kernel_v: L
>assignvariableop_60_adam_feature_maps_feature_map_conv1_bias_v: M
?assignvariableop_61_adam_feature_maps_feature_map_norm1_gamma_v: L
>assignvariableop_62_adam_feature_maps_feature_map_norm1_beta_v: Z
@assignvariableop_63_adam_feature_maps_feature_map_conv2_kernel_v: @L
>assignvariableop_64_adam_feature_maps_feature_map_conv2_bias_v:@M
?assignvariableop_65_adam_feature_maps_feature_map_norm2_gamma_v:@L
>assignvariableop_66_adam_feature_maps_feature_map_norm2_beta_v:@Z
@assignvariableop_67_adam_feature_maps_feature_map_conv3_kernel_v:@@L
>assignvariableop_68_adam_feature_maps_feature_map_conv3_bias_v:@M
?assignvariableop_69_adam_feature_maps_feature_map_norm3_gamma_v:@L
>assignvariableop_70_adam_feature_maps_feature_map_norm3_beta_v:@[
@assignvariableop_71_adam_feature_maps_feature_map_conv4_kernel_v:@?M
>assignvariableop_72_adam_feature_maps_feature_map_conv4_bias_v:	?N
?assignvariableop_73_adam_feature_maps_feature_map_norm4_gamma_v:	?M
>assignvariableop_74_adam_feature_maps_feature_map_norm4_beta_v:	?[
@assignvariableop_75_adam_primary_caps_primary_cap_dconv_kernel_v:		?M
>assignvariableop_76_adam_primary_caps_primary_cap_dconv_bias_v:	?
identity_78??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_8?AssignVariableOp_9?$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*?#
value?#B?#NBKlayer_with_weights-2/digit_caps_transform_tensor/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-2/digit_caps_log_priors/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBglayer_with_weights-2/digit_caps_transform_tensor/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBalayer_with_weights-2/digit_caps_log_priors/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBglayer_with_weights-2/digit_caps_transform_tensor/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBalayer_with_weights-2/digit_caps_log_priors/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*?
value?B?NB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*\
dtypesR
P2N	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp7assignvariableop_digit_caps_digit_caps_transform_tensorIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp3assignvariableop_1_digit_caps_digit_caps_log_priorsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp8assignvariableop_7_feature_maps_feature_map_conv1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp6assignvariableop_8_feature_maps_feature_map_conv1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp7assignvariableop_9_feature_maps_feature_map_norm1_gammaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp7assignvariableop_10_feature_maps_feature_map_norm1_betaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp9assignvariableop_11_feature_maps_feature_map_conv2_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp7assignvariableop_12_feature_maps_feature_map_conv2_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp8assignvariableop_13_feature_maps_feature_map_norm2_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp7assignvariableop_14_feature_maps_feature_map_norm2_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp9assignvariableop_15_feature_maps_feature_map_conv3_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp7assignvariableop_16_feature_maps_feature_map_conv3_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp8assignvariableop_17_feature_maps_feature_map_norm3_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp7assignvariableop_18_feature_maps_feature_map_norm3_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp9assignvariableop_19_feature_maps_feature_map_conv4_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp7assignvariableop_20_feature_maps_feature_map_conv4_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp8assignvariableop_21_feature_maps_feature_map_norm4_gammaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp7assignvariableop_22_feature_maps_feature_map_norm4_betaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp>assignvariableop_23_feature_maps_feature_map_norm1_moving_meanIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpBassignvariableop_24_feature_maps_feature_map_norm1_moving_varianceIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp>assignvariableop_25_feature_maps_feature_map_norm2_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpBassignvariableop_26_feature_maps_feature_map_norm2_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp>assignvariableop_27_feature_maps_feature_map_norm3_moving_meanIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpBassignvariableop_28_feature_maps_feature_map_norm3_moving_varianceIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp>assignvariableop_29_feature_maps_feature_map_norm4_moving_meanIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpBassignvariableop_30_feature_maps_feature_map_norm4_moving_varianceIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp9assignvariableop_31_primary_caps_primary_cap_dconv_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp7assignvariableop_32_primary_caps_primary_cap_dconv_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpassignvariableop_33_totalIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOpassignvariableop_34_countIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOpassignvariableop_35_total_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOpassignvariableop_36_count_1Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOpAassignvariableop_37_adam_digit_caps_digit_caps_transform_tensor_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp;assignvariableop_38_adam_digit_caps_digit_caps_log_priors_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp@assignvariableop_39_adam_feature_maps_feature_map_conv1_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp>assignvariableop_40_adam_feature_maps_feature_map_conv1_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp?assignvariableop_41_adam_feature_maps_feature_map_norm1_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp>assignvariableop_42_adam_feature_maps_feature_map_norm1_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp@assignvariableop_43_adam_feature_maps_feature_map_conv2_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp>assignvariableop_44_adam_feature_maps_feature_map_conv2_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp?assignvariableop_45_adam_feature_maps_feature_map_norm2_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp>assignvariableop_46_adam_feature_maps_feature_map_norm2_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp@assignvariableop_47_adam_feature_maps_feature_map_conv3_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp>assignvariableop_48_adam_feature_maps_feature_map_conv3_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp?assignvariableop_49_adam_feature_maps_feature_map_norm3_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp>assignvariableop_50_adam_feature_maps_feature_map_norm3_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp@assignvariableop_51_adam_feature_maps_feature_map_conv4_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp>assignvariableop_52_adam_feature_maps_feature_map_conv4_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp?assignvariableop_53_adam_feature_maps_feature_map_norm4_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp>assignvariableop_54_adam_feature_maps_feature_map_norm4_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp@assignvariableop_55_adam_primary_caps_primary_cap_dconv_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp>assignvariableop_56_adam_primary_caps_primary_cap_dconv_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOpAassignvariableop_57_adam_digit_caps_digit_caps_transform_tensor_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp;assignvariableop_58_adam_digit_caps_digit_caps_log_priors_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp@assignvariableop_59_adam_feature_maps_feature_map_conv1_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp>assignvariableop_60_adam_feature_maps_feature_map_conv1_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp?assignvariableop_61_adam_feature_maps_feature_map_norm1_gamma_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp>assignvariableop_62_adam_feature_maps_feature_map_norm1_beta_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp@assignvariableop_63_adam_feature_maps_feature_map_conv2_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp>assignvariableop_64_adam_feature_maps_feature_map_conv2_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp?assignvariableop_65_adam_feature_maps_feature_map_norm2_gamma_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp>assignvariableop_66_adam_feature_maps_feature_map_norm2_beta_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp@assignvariableop_67_adam_feature_maps_feature_map_conv3_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp>assignvariableop_68_adam_feature_maps_feature_map_conv3_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp?assignvariableop_69_adam_feature_maps_feature_map_norm3_gamma_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp>assignvariableop_70_adam_feature_maps_feature_map_norm3_beta_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp@assignvariableop_71_adam_feature_maps_feature_map_conv4_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp>assignvariableop_72_adam_feature_maps_feature_map_conv4_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp?assignvariableop_73_adam_feature_maps_feature_map_norm4_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp>assignvariableop_74_adam_feature_maps_feature_map_norm4_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp@assignvariableop_75_adam_primary_caps_primary_cap_dconv_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOp>assignvariableop_76_adam_primary_caps_primary_cap_dconv_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_77Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_78IdentityIdentity_77:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_78Identity_78:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
??
?
H__inference_feature_maps_layer_call_and_return_conditional_losses_725224
input_imagesJ
0feature_map_conv1_conv2d_readvariableop_resource: ?
1feature_map_conv1_biasadd_readvariableop_resource: 7
)feature_map_norm1_readvariableop_resource: 9
+feature_map_norm1_readvariableop_1_resource: H
:feature_map_norm1_fusedbatchnormv3_readvariableop_resource: J
<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource: J
0feature_map_conv2_conv2d_readvariableop_resource: @?
1feature_map_conv2_biasadd_readvariableop_resource:@7
)feature_map_norm2_readvariableop_resource:@9
+feature_map_norm2_readvariableop_1_resource:@H
:feature_map_norm2_fusedbatchnormv3_readvariableop_resource:@J
<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource:@J
0feature_map_conv3_conv2d_readvariableop_resource:@@?
1feature_map_conv3_biasadd_readvariableop_resource:@7
)feature_map_norm3_readvariableop_resource:@9
+feature_map_norm3_readvariableop_1_resource:@H
:feature_map_norm3_fusedbatchnormv3_readvariableop_resource:@J
<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource:@K
0feature_map_conv4_conv2d_readvariableop_resource:@?@
1feature_map_conv4_biasadd_readvariableop_resource:	?8
)feature_map_norm4_readvariableop_resource:	?:
+feature_map_norm4_readvariableop_1_resource:	?I
:feature_map_norm4_fusedbatchnormv3_readvariableop_resource:	?K
<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource:	?
identity??(feature_map_conv1/BiasAdd/ReadVariableOp?'feature_map_conv1/Conv2D/ReadVariableOp?(feature_map_conv2/BiasAdd/ReadVariableOp?'feature_map_conv2/Conv2D/ReadVariableOp?(feature_map_conv3/BiasAdd/ReadVariableOp?'feature_map_conv3/Conv2D/ReadVariableOp?(feature_map_conv4/BiasAdd/ReadVariableOp?'feature_map_conv4/Conv2D/ReadVariableOp? feature_map_norm1/AssignNewValue?"feature_map_norm1/AssignNewValue_1?1feature_map_norm1/FusedBatchNormV3/ReadVariableOp?3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1? feature_map_norm1/ReadVariableOp?"feature_map_norm1/ReadVariableOp_1? feature_map_norm2/AssignNewValue?"feature_map_norm2/AssignNewValue_1?1feature_map_norm2/FusedBatchNormV3/ReadVariableOp?3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1? feature_map_norm2/ReadVariableOp?"feature_map_norm2/ReadVariableOp_1? feature_map_norm3/AssignNewValue?"feature_map_norm3/AssignNewValue_1?1feature_map_norm3/FusedBatchNormV3/ReadVariableOp?3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1? feature_map_norm3/ReadVariableOp?"feature_map_norm3/ReadVariableOp_1? feature_map_norm4/AssignNewValue?"feature_map_norm4/AssignNewValue_1?1feature_map_norm4/FusedBatchNormV3/ReadVariableOp?3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1? feature_map_norm4/ReadVariableOp?"feature_map_norm4/ReadVariableOp_1?
'feature_map_conv1/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
feature_map_conv1/Conv2DConv2Dinput_images/feature_map_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
(feature_map_conv1/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
feature_map_conv1/BiasAddBiasAdd!feature_map_conv1/Conv2D:output:00feature_map_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? |
feature_map_conv1/ReluRelu"feature_map_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
 feature_map_norm1/ReadVariableOpReadVariableOp)feature_map_norm1_readvariableop_resource*
_output_shapes
: *
dtype0?
"feature_map_norm1/ReadVariableOp_1ReadVariableOp+feature_map_norm1_readvariableop_1_resource*
_output_shapes
: *
dtype0?
1feature_map_norm1/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
"feature_map_norm1/FusedBatchNormV3FusedBatchNormV3$feature_map_conv1/Relu:activations:0(feature_map_norm1/ReadVariableOp:value:0*feature_map_norm1/ReadVariableOp_1:value:09feature_map_norm1/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
 feature_map_norm1/AssignNewValueAssignVariableOp:feature_map_norm1_fusedbatchnormv3_readvariableop_resource/feature_map_norm1/FusedBatchNormV3:batch_mean:02^feature_map_norm1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
"feature_map_norm1/AssignNewValue_1AssignVariableOp<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm1/FusedBatchNormV3:batch_variance:04^feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
'feature_map_conv2/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
feature_map_conv2/Conv2DConv2D&feature_map_norm1/FusedBatchNormV3:y:0/feature_map_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
(feature_map_conv2/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
feature_map_conv2/BiasAddBiasAdd!feature_map_conv2/Conv2D:output:00feature_map_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@|
feature_map_conv2/ReluRelu"feature_map_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
 feature_map_norm2/ReadVariableOpReadVariableOp)feature_map_norm2_readvariableop_resource*
_output_shapes
:@*
dtype0?
"feature_map_norm2/ReadVariableOp_1ReadVariableOp+feature_map_norm2_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
1feature_map_norm2/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
"feature_map_norm2/FusedBatchNormV3FusedBatchNormV3$feature_map_conv2/Relu:activations:0(feature_map_norm2/ReadVariableOp:value:0*feature_map_norm2/ReadVariableOp_1:value:09feature_map_norm2/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
 feature_map_norm2/AssignNewValueAssignVariableOp:feature_map_norm2_fusedbatchnormv3_readvariableop_resource/feature_map_norm2/FusedBatchNormV3:batch_mean:02^feature_map_norm2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
"feature_map_norm2/AssignNewValue_1AssignVariableOp<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm2/FusedBatchNormV3:batch_variance:04^feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
'feature_map_conv3/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
feature_map_conv3/Conv2DConv2D&feature_map_norm2/FusedBatchNormV3:y:0/feature_map_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
(feature_map_conv3/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
feature_map_conv3/BiasAddBiasAdd!feature_map_conv3/Conv2D:output:00feature_map_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@|
feature_map_conv3/ReluRelu"feature_map_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
 feature_map_norm3/ReadVariableOpReadVariableOp)feature_map_norm3_readvariableop_resource*
_output_shapes
:@*
dtype0?
"feature_map_norm3/ReadVariableOp_1ReadVariableOp+feature_map_norm3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
1feature_map_norm3/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
"feature_map_norm3/FusedBatchNormV3FusedBatchNormV3$feature_map_conv3/Relu:activations:0(feature_map_norm3/ReadVariableOp:value:0*feature_map_norm3/ReadVariableOp_1:value:09feature_map_norm3/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
 feature_map_norm3/AssignNewValueAssignVariableOp:feature_map_norm3_fusedbatchnormv3_readvariableop_resource/feature_map_norm3/FusedBatchNormV3:batch_mean:02^feature_map_norm3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
"feature_map_norm3/AssignNewValue_1AssignVariableOp<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm3/FusedBatchNormV3:batch_variance:04^feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
'feature_map_conv4/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
feature_map_conv4/Conv2DConv2D&feature_map_norm3/FusedBatchNormV3:y:0/feature_map_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
?
(feature_map_conv4/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
feature_map_conv4/BiasAddBiasAdd!feature_map_conv4/Conv2D:output:00feature_map_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?}
feature_map_conv4/ReluRelu"feature_map_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		??
 feature_map_norm4/ReadVariableOpReadVariableOp)feature_map_norm4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"feature_map_norm4/ReadVariableOp_1ReadVariableOp+feature_map_norm4_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
1feature_map_norm4/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
"feature_map_norm4/FusedBatchNormV3FusedBatchNormV3$feature_map_conv4/Relu:activations:0(feature_map_norm4/ReadVariableOp:value:0*feature_map_norm4/ReadVariableOp_1:value:09feature_map_norm4/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????		?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
 feature_map_norm4/AssignNewValueAssignVariableOp:feature_map_norm4_fusedbatchnormv3_readvariableop_resource/feature_map_norm4/FusedBatchNormV3:batch_mean:02^feature_map_norm4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
"feature_map_norm4/AssignNewValue_1AssignVariableOp<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm4/FusedBatchNormV3:batch_variance:04^feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentity&feature_map_norm4/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????		??
NoOpNoOp)^feature_map_conv1/BiasAdd/ReadVariableOp(^feature_map_conv1/Conv2D/ReadVariableOp)^feature_map_conv2/BiasAdd/ReadVariableOp(^feature_map_conv2/Conv2D/ReadVariableOp)^feature_map_conv3/BiasAdd/ReadVariableOp(^feature_map_conv3/Conv2D/ReadVariableOp)^feature_map_conv4/BiasAdd/ReadVariableOp(^feature_map_conv4/Conv2D/ReadVariableOp!^feature_map_norm1/AssignNewValue#^feature_map_norm1/AssignNewValue_12^feature_map_norm1/FusedBatchNormV3/ReadVariableOp4^feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm1/ReadVariableOp#^feature_map_norm1/ReadVariableOp_1!^feature_map_norm2/AssignNewValue#^feature_map_norm2/AssignNewValue_12^feature_map_norm2/FusedBatchNormV3/ReadVariableOp4^feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm2/ReadVariableOp#^feature_map_norm2/ReadVariableOp_1!^feature_map_norm3/AssignNewValue#^feature_map_norm3/AssignNewValue_12^feature_map_norm3/FusedBatchNormV3/ReadVariableOp4^feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm3/ReadVariableOp#^feature_map_norm3/ReadVariableOp_1!^feature_map_norm4/AssignNewValue#^feature_map_norm4/AssignNewValue_12^feature_map_norm4/FusedBatchNormV3/ReadVariableOp4^feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm4/ReadVariableOp#^feature_map_norm4/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????: : : : : : : : : : : : : : : : : : : : : : : : 2T
(feature_map_conv1/BiasAdd/ReadVariableOp(feature_map_conv1/BiasAdd/ReadVariableOp2R
'feature_map_conv1/Conv2D/ReadVariableOp'feature_map_conv1/Conv2D/ReadVariableOp2T
(feature_map_conv2/BiasAdd/ReadVariableOp(feature_map_conv2/BiasAdd/ReadVariableOp2R
'feature_map_conv2/Conv2D/ReadVariableOp'feature_map_conv2/Conv2D/ReadVariableOp2T
(feature_map_conv3/BiasAdd/ReadVariableOp(feature_map_conv3/BiasAdd/ReadVariableOp2R
'feature_map_conv3/Conv2D/ReadVariableOp'feature_map_conv3/Conv2D/ReadVariableOp2T
(feature_map_conv4/BiasAdd/ReadVariableOp(feature_map_conv4/BiasAdd/ReadVariableOp2R
'feature_map_conv4/Conv2D/ReadVariableOp'feature_map_conv4/Conv2D/ReadVariableOp2D
 feature_map_norm1/AssignNewValue feature_map_norm1/AssignNewValue2H
"feature_map_norm1/AssignNewValue_1"feature_map_norm1/AssignNewValue_12f
1feature_map_norm1/FusedBatchNormV3/ReadVariableOp1feature_map_norm1/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_13feature_map_norm1/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm1/ReadVariableOp feature_map_norm1/ReadVariableOp2H
"feature_map_norm1/ReadVariableOp_1"feature_map_norm1/ReadVariableOp_12D
 feature_map_norm2/AssignNewValue feature_map_norm2/AssignNewValue2H
"feature_map_norm2/AssignNewValue_1"feature_map_norm2/AssignNewValue_12f
1feature_map_norm2/FusedBatchNormV3/ReadVariableOp1feature_map_norm2/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_13feature_map_norm2/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm2/ReadVariableOp feature_map_norm2/ReadVariableOp2H
"feature_map_norm2/ReadVariableOp_1"feature_map_norm2/ReadVariableOp_12D
 feature_map_norm3/AssignNewValue feature_map_norm3/AssignNewValue2H
"feature_map_norm3/AssignNewValue_1"feature_map_norm3/AssignNewValue_12f
1feature_map_norm3/FusedBatchNormV3/ReadVariableOp1feature_map_norm3/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_13feature_map_norm3/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm3/ReadVariableOp feature_map_norm3/ReadVariableOp2H
"feature_map_norm3/ReadVariableOp_1"feature_map_norm3/ReadVariableOp_12D
 feature_map_norm4/AssignNewValue feature_map_norm4/AssignNewValue2H
"feature_map_norm4/AssignNewValue_1"feature_map_norm4/AssignNewValue_12f
1feature_map_norm4/FusedBatchNormV3/ReadVariableOp1feature_map_norm4/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_13feature_map_norm4/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm4/ReadVariableOp feature_map_norm4/ReadVariableOp2H
"feature_map_norm4/ReadVariableOp_1"feature_map_norm4/ReadVariableOp_1:] Y
/
_output_shapes
:?????????
&
_user_specified_nameinput_images
?
?
map_while_body_724852$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1#
map_while_map_strided_slice_1_0_
[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0D
*map_while_matmul_readvariableop_resource_0:

map_while_identity
map_while_identity_1
map_while_identity_2
map_while_identity_3!
map_while_map_strided_slice_1]
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensorB
(map_while_matmul_readvariableop_resource:
??map/while/MatMul/ReadVariableOp?
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            ?
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholderDmap/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*&
_output_shapes
:
*
element_dtype0?
map/while/MatMul/ReadVariableOpReadVariableOp*map_while_matmul_readvariableop_resource_0*&
_output_shapes
:
*
dtype0?
map/while/MatMulBatchMatMulV2'map/while/MatMul/ReadVariableOp:value:04map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*&
_output_shapes
:
?
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholdermap/while/MatMul:output:0*
_output_shapes
: *
element_dtype0:???Q
map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
map/while/addAddV2map_while_placeholdermap/while/add/y:output:0*
T0*
_output_shapes
: S
map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
map/while/add_1AddV2 map_while_map_while_loop_countermap/while/add_1/y:output:0*
T0*
_output_shapes
: e
map/while/IdentityIdentitymap/while/add_1:z:0^map/while/NoOp*
T0*
_output_shapes
: o
map/while/Identity_1Identitymap_while_map_strided_slice^map/while/NoOp*
T0*
_output_shapes
: e
map/while/Identity_2Identitymap/while/add:z:0^map/while/NoOp*
T0*
_output_shapes
: ?
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^map/while/NoOp*
T0*
_output_shapes
: :???r
map/while/NoOpNoOp ^map/while/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"V
(map_while_matmul_readvariableop_resource*map_while_matmul_readvariableop_resource_0"?
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : : : 2B
map/while/MatMul/ReadVariableOpmap/while/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?&
!__inference__wrapped_model_724390
input_imagesi
Oefficient_capsnet_feature_maps_feature_map_conv1_conv2d_readvariableop_resource: ^
Pefficient_capsnet_feature_maps_feature_map_conv1_biasadd_readvariableop_resource: V
Hefficient_capsnet_feature_maps_feature_map_norm1_readvariableop_resource: X
Jefficient_capsnet_feature_maps_feature_map_norm1_readvariableop_1_resource: g
Yefficient_capsnet_feature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_resource: i
[efficient_capsnet_feature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource: i
Oefficient_capsnet_feature_maps_feature_map_conv2_conv2d_readvariableop_resource: @^
Pefficient_capsnet_feature_maps_feature_map_conv2_biasadd_readvariableop_resource:@V
Hefficient_capsnet_feature_maps_feature_map_norm2_readvariableop_resource:@X
Jefficient_capsnet_feature_maps_feature_map_norm2_readvariableop_1_resource:@g
Yefficient_capsnet_feature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_resource:@i
[efficient_capsnet_feature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource:@i
Oefficient_capsnet_feature_maps_feature_map_conv3_conv2d_readvariableop_resource:@@^
Pefficient_capsnet_feature_maps_feature_map_conv3_biasadd_readvariableop_resource:@V
Hefficient_capsnet_feature_maps_feature_map_norm3_readvariableop_resource:@X
Jefficient_capsnet_feature_maps_feature_map_norm3_readvariableop_1_resource:@g
Yefficient_capsnet_feature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_resource:@i
[efficient_capsnet_feature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource:@j
Oefficient_capsnet_feature_maps_feature_map_conv4_conv2d_readvariableop_resource:@?_
Pefficient_capsnet_feature_maps_feature_map_conv4_biasadd_readvariableop_resource:	?W
Hefficient_capsnet_feature_maps_feature_map_norm4_readvariableop_resource:	?Y
Jefficient_capsnet_feature_maps_feature_map_norm4_readvariableop_1_resource:	?h
Yefficient_capsnet_feature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_resource:	?j
[efficient_capsnet_feature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource:	?j
Oefficient_capsnet_primary_caps_primary_cap_dconv_conv2d_readvariableop_resource:		?_
Pefficient_capsnet_primary_caps_primary_cap_dconv_biasadd_readvariableop_resource:	?H
.efficient_capsnet_digit_caps_map_while_input_6:
&
"efficient_capsnet_digit_caps_mul_xP
:efficient_capsnet_digit_caps_add_3_readvariableop_resource:

identity??1Efficient-CapsNet/digit_caps/add_3/ReadVariableOp?&Efficient-CapsNet/digit_caps/map/while?GEfficient-CapsNet/feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp?FEfficient-CapsNet/feature_maps/feature_map_conv1/Conv2D/ReadVariableOp?GEfficient-CapsNet/feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp?FEfficient-CapsNet/feature_maps/feature_map_conv2/Conv2D/ReadVariableOp?GEfficient-CapsNet/feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp?FEfficient-CapsNet/feature_maps/feature_map_conv3/Conv2D/ReadVariableOp?GEfficient-CapsNet/feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp?FEfficient-CapsNet/feature_maps/feature_map_conv4/Conv2D/ReadVariableOp?PEfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp?REfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1??Efficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOp?AEfficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOp_1?PEfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp?REfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1??Efficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOp?AEfficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOp_1?PEfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp?REfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1??Efficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOp?AEfficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOp_1?PEfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp?REfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1??Efficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOp?AEfficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOp_1?GEfficient-CapsNet/primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp?FEfficient-CapsNet/primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp?
FEfficient-CapsNet/feature_maps/feature_map_conv1/Conv2D/ReadVariableOpReadVariableOpOefficient_capsnet_feature_maps_feature_map_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
7Efficient-CapsNet/feature_maps/feature_map_conv1/Conv2DConv2Dinput_imagesNEfficient-CapsNet/feature_maps/feature_map_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
GEfficient-CapsNet/feature_maps/feature_map_conv1/BiasAdd/ReadVariableOpReadVariableOpPefficient_capsnet_feature_maps_feature_map_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
8Efficient-CapsNet/feature_maps/feature_map_conv1/BiasAddBiasAdd@Efficient-CapsNet/feature_maps/feature_map_conv1/Conv2D:output:0OEfficient-CapsNet/feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
5Efficient-CapsNet/feature_maps/feature_map_conv1/ReluReluAEfficient-CapsNet/feature_maps/feature_map_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
?Efficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOpReadVariableOpHefficient_capsnet_feature_maps_feature_map_norm1_readvariableop_resource*
_output_shapes
: *
dtype0?
AEfficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOp_1ReadVariableOpJefficient_capsnet_feature_maps_feature_map_norm1_readvariableop_1_resource*
_output_shapes
: *
dtype0?
PEfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOpReadVariableOpYefficient_capsnet_feature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
REfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp[efficient_capsnet_feature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
AEfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3FusedBatchNormV3CEfficient-CapsNet/feature_maps/feature_map_conv1/Relu:activations:0GEfficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOp:value:0IEfficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOp_1:value:0XEfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp:value:0ZEfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
FEfficient-CapsNet/feature_maps/feature_map_conv2/Conv2D/ReadVariableOpReadVariableOpOefficient_capsnet_feature_maps_feature_map_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
7Efficient-CapsNet/feature_maps/feature_map_conv2/Conv2DConv2DEEfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3:y:0NEfficient-CapsNet/feature_maps/feature_map_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
GEfficient-CapsNet/feature_maps/feature_map_conv2/BiasAdd/ReadVariableOpReadVariableOpPefficient_capsnet_feature_maps_feature_map_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
8Efficient-CapsNet/feature_maps/feature_map_conv2/BiasAddBiasAdd@Efficient-CapsNet/feature_maps/feature_map_conv2/Conv2D:output:0OEfficient-CapsNet/feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
5Efficient-CapsNet/feature_maps/feature_map_conv2/ReluReluAEfficient-CapsNet/feature_maps/feature_map_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
?Efficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOpReadVariableOpHefficient_capsnet_feature_maps_feature_map_norm2_readvariableop_resource*
_output_shapes
:@*
dtype0?
AEfficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOp_1ReadVariableOpJefficient_capsnet_feature_maps_feature_map_norm2_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
PEfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOpReadVariableOpYefficient_capsnet_feature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
REfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp[efficient_capsnet_feature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
AEfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3FusedBatchNormV3CEfficient-CapsNet/feature_maps/feature_map_conv2/Relu:activations:0GEfficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOp:value:0IEfficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOp_1:value:0XEfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp:value:0ZEfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
FEfficient-CapsNet/feature_maps/feature_map_conv3/Conv2D/ReadVariableOpReadVariableOpOefficient_capsnet_feature_maps_feature_map_conv3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
7Efficient-CapsNet/feature_maps/feature_map_conv3/Conv2DConv2DEEfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3:y:0NEfficient-CapsNet/feature_maps/feature_map_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
GEfficient-CapsNet/feature_maps/feature_map_conv3/BiasAdd/ReadVariableOpReadVariableOpPefficient_capsnet_feature_maps_feature_map_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
8Efficient-CapsNet/feature_maps/feature_map_conv3/BiasAddBiasAdd@Efficient-CapsNet/feature_maps/feature_map_conv3/Conv2D:output:0OEfficient-CapsNet/feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
5Efficient-CapsNet/feature_maps/feature_map_conv3/ReluReluAEfficient-CapsNet/feature_maps/feature_map_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
?Efficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOpReadVariableOpHefficient_capsnet_feature_maps_feature_map_norm3_readvariableop_resource*
_output_shapes
:@*
dtype0?
AEfficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOp_1ReadVariableOpJefficient_capsnet_feature_maps_feature_map_norm3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
PEfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOpReadVariableOpYefficient_capsnet_feature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
REfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp[efficient_capsnet_feature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
AEfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3FusedBatchNormV3CEfficient-CapsNet/feature_maps/feature_map_conv3/Relu:activations:0GEfficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOp:value:0IEfficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOp_1:value:0XEfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp:value:0ZEfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
FEfficient-CapsNet/feature_maps/feature_map_conv4/Conv2D/ReadVariableOpReadVariableOpOefficient_capsnet_feature_maps_feature_map_conv4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
7Efficient-CapsNet/feature_maps/feature_map_conv4/Conv2DConv2DEEfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3:y:0NEfficient-CapsNet/feature_maps/feature_map_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
?
GEfficient-CapsNet/feature_maps/feature_map_conv4/BiasAdd/ReadVariableOpReadVariableOpPefficient_capsnet_feature_maps_feature_map_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8Efficient-CapsNet/feature_maps/feature_map_conv4/BiasAddBiasAdd@Efficient-CapsNet/feature_maps/feature_map_conv4/Conv2D:output:0OEfficient-CapsNet/feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		??
5Efficient-CapsNet/feature_maps/feature_map_conv4/ReluReluAEfficient-CapsNet/feature_maps/feature_map_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		??
?Efficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOpReadVariableOpHefficient_capsnet_feature_maps_feature_map_norm4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AEfficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOp_1ReadVariableOpJefficient_capsnet_feature_maps_feature_map_norm4_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
PEfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOpReadVariableOpYefficient_capsnet_feature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
REfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp[efficient_capsnet_feature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
AEfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3FusedBatchNormV3CEfficient-CapsNet/feature_maps/feature_map_conv4/Relu:activations:0GEfficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOp:value:0IEfficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOp_1:value:0XEfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp:value:0ZEfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????		?:?:?:?:?:*
epsilon%o?:*
is_training( ?
FEfficient-CapsNet/primary_caps/primary_cap_dconv/Conv2D/ReadVariableOpReadVariableOpOefficient_capsnet_primary_caps_primary_cap_dconv_conv2d_readvariableop_resource*'
_output_shapes
:		?*
dtype0?
7Efficient-CapsNet/primary_caps/primary_cap_dconv/Conv2DConv2DEEfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3:y:0NEfficient-CapsNet/primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
GEfficient-CapsNet/primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOpReadVariableOpPefficient_capsnet_primary_caps_primary_cap_dconv_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8Efficient-CapsNet/primary_caps/primary_cap_dconv/BiasAddBiasAdd@Efficient-CapsNet/primary_caps/primary_cap_dconv/Conv2D:output:0OEfficient-CapsNet/primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
5Efficient-CapsNet/primary_caps/primary_cap_dconv/ReluReluAEfficient-CapsNet/primary_caps/primary_cap_dconv/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
8Efficient-CapsNet/primary_caps/primary_cap_reshape/ShapeShapeCEfficient-CapsNet/primary_caps/primary_cap_dconv/Relu:activations:0*
T0*
_output_shapes
:?
FEfficient-CapsNet/primary_caps/primary_cap_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
HEfficient-CapsNet/primary_caps/primary_cap_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
HEfficient-CapsNet/primary_caps/primary_cap_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
@Efficient-CapsNet/primary_caps/primary_cap_reshape/strided_sliceStridedSliceAEfficient-CapsNet/primary_caps/primary_cap_reshape/Shape:output:0OEfficient-CapsNet/primary_caps/primary_cap_reshape/strided_slice/stack:output:0QEfficient-CapsNet/primary_caps/primary_cap_reshape/strided_slice/stack_1:output:0QEfficient-CapsNet/primary_caps/primary_cap_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
BEfficient-CapsNet/primary_caps/primary_cap_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
??????????
BEfficient-CapsNet/primary_caps/primary_cap_reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
@Efficient-CapsNet/primary_caps/primary_cap_reshape/Reshape/shapePackIEfficient-CapsNet/primary_caps/primary_cap_reshape/strided_slice:output:0KEfficient-CapsNet/primary_caps/primary_cap_reshape/Reshape/shape/1:output:0KEfficient-CapsNet/primary_caps/primary_cap_reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
:Efficient-CapsNet/primary_caps/primary_cap_reshape/ReshapeReshapeCEfficient-CapsNet/primary_caps/primary_cap_dconv/Relu:activations:0IEfficient-CapsNet/primary_caps/primary_cap_reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:??????????
:Efficient-CapsNet/primary_caps/primary_cap_squash/norm/mulMulCEfficient-CapsNet/primary_caps/primary_cap_reshape/Reshape:output:0CEfficient-CapsNet/primary_caps/primary_cap_reshape/Reshape:output:0*
T0*+
_output_shapes
:??????????
LEfficient-CapsNet/primary_caps/primary_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
:Efficient-CapsNet/primary_caps/primary_cap_squash/norm/SumSum>Efficient-CapsNet/primary_caps/primary_cap_squash/norm/mul:z:0UEfficient-CapsNet/primary_caps/primary_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(?
;Efficient-CapsNet/primary_caps/primary_cap_squash/norm/SqrtSqrtCEfficient-CapsNet/primary_caps/primary_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:??????????
5Efficient-CapsNet/primary_caps/primary_cap_squash/ExpExp?Efficient-CapsNet/primary_caps/primary_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:??????????
;Efficient-CapsNet/primary_caps/primary_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
9Efficient-CapsNet/primary_caps/primary_cap_squash/truedivRealDivDEfficient-CapsNet/primary_caps/primary_cap_squash/truediv/x:output:09Efficient-CapsNet/primary_caps/primary_cap_squash/Exp:y:0*
T0*+
_output_shapes
:?????????|
7Efficient-CapsNet/primary_caps/primary_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
5Efficient-CapsNet/primary_caps/primary_cap_squash/subSub@Efficient-CapsNet/primary_caps/primary_cap_squash/sub/x:output:0=Efficient-CapsNet/primary_caps/primary_cap_squash/truediv:z:0*
T0*+
_output_shapes
:?????????|
7Efficient-CapsNet/primary_caps/primary_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
5Efficient-CapsNet/primary_caps/primary_cap_squash/addAddV2?Efficient-CapsNet/primary_caps/primary_cap_squash/norm/Sqrt:y:0@Efficient-CapsNet/primary_caps/primary_cap_squash/add/y:output:0*
T0*+
_output_shapes
:??????????
;Efficient-CapsNet/primary_caps/primary_cap_squash/truediv_1RealDivCEfficient-CapsNet/primary_caps/primary_cap_reshape/Reshape:output:09Efficient-CapsNet/primary_caps/primary_cap_squash/add:z:0*
T0*+
_output_shapes
:??????????
5Efficient-CapsNet/primary_caps/primary_cap_squash/mulMul9Efficient-CapsNet/primary_caps/primary_cap_squash/sub:z:0?Efficient-CapsNet/primary_caps/primary_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:?????????m
+Efficient-CapsNet/digit_caps/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
'Efficient-CapsNet/digit_caps/ExpandDims
ExpandDims9Efficient-CapsNet/primary_caps/primary_cap_squash/mul:z:04Efficient-CapsNet/digit_caps/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
+Efficient-CapsNet/digit_caps/Tile/multiplesConst*
_output_shapes
:*
dtype0*%
valueB"   
         ?
!Efficient-CapsNet/digit_caps/TileTile0Efficient-CapsNet/digit_caps/ExpandDims:output:04Efficient-CapsNet/digit_caps/Tile/multiples:output:0*
T0*/
_output_shapes
:?????????
|
1Efficient-CapsNet/digit_caps/digit_cap_inputs/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
-Efficient-CapsNet/digit_caps/digit_cap_inputs
ExpandDims*Efficient-CapsNet/digit_caps/Tile:output:0:Efficient-CapsNet/digit_caps/digit_cap_inputs/dim:output:0*
T0*3
_output_shapes!
:?????????
?
&Efficient-CapsNet/digit_caps/map/ShapeShape6Efficient-CapsNet/digit_caps/digit_cap_inputs:output:0*
T0*
_output_shapes
:~
4Efficient-CapsNet/digit_caps/map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6Efficient-CapsNet/digit_caps/map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6Efficient-CapsNet/digit_caps/map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.Efficient-CapsNet/digit_caps/map/strided_sliceStridedSlice/Efficient-CapsNet/digit_caps/map/Shape:output:0=Efficient-CapsNet/digit_caps/map/strided_slice/stack:output:0?Efficient-CapsNet/digit_caps/map/strided_slice/stack_1:output:0?Efficient-CapsNet/digit_caps/map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
<Efficient-CapsNet/digit_caps/map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.Efficient-CapsNet/digit_caps/map/TensorArrayV2TensorListReserveEEfficient-CapsNet/digit_caps/map/TensorArrayV2/element_shape:output:07Efficient-CapsNet/digit_caps/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
VEfficient-CapsNet/digit_caps/map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            ?
HEfficient-CapsNet/digit_caps/map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor6Efficient-CapsNet/digit_caps/digit_cap_inputs:output:0_Efficient-CapsNet/digit_caps/map/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???h
&Efficient-CapsNet/digit_caps/map/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
>Efficient-CapsNet/digit_caps/map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
0Efficient-CapsNet/digit_caps/map/TensorArrayV2_1TensorListReserveGEfficient-CapsNet/digit_caps/map/TensorArrayV2_1/element_shape:output:07Efficient-CapsNet/digit_caps/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???u
3Efficient-CapsNet/digit_caps/map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
&Efficient-CapsNet/digit_caps/map/whileWhile<Efficient-CapsNet/digit_caps/map/while/loop_counter:output:07Efficient-CapsNet/digit_caps/map/strided_slice:output:0/Efficient-CapsNet/digit_caps/map/Const:output:09Efficient-CapsNet/digit_caps/map/TensorArrayV2_1:handle:07Efficient-CapsNet/digit_caps/map/strided_slice:output:0XEfficient-CapsNet/digit_caps/map/TensorArrayUnstack/TensorListFromTensor:output_handle:0.efficient_capsnet_digit_caps_map_while_input_6*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : *#
_read_only_resource_inputs
*
_stateful_parallelism( *>
body6R4
2Efficient-CapsNet_digit_caps_map_while_body_724280*>
cond6R4
2Efficient-CapsNet_digit_caps_map_while_cond_724279*!
output_shapes
: : : : : : : ?
QEfficient-CapsNet/digit_caps/map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            ?
CEfficient-CapsNet/digit_caps/map/TensorArrayV2Stack/TensorListStackTensorListStack/Efficient-CapsNet/digit_caps/map/while:output:3ZEfficient-CapsNet/digit_caps/map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:?????????
*
element_dtype0?
2Efficient-CapsNet/digit_caps/digit_cap_predictionsSqueezeLEfficient-CapsNet/digit_caps/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*/
_output_shapes
:?????????
*
squeeze_dims

??????????
1Efficient-CapsNet/digit_caps/digit_cap_attentionsBatchMatMulV2;Efficient-CapsNet/digit_caps/digit_cap_predictions:output:0;Efficient-CapsNet/digit_caps/digit_cap_predictions:output:0*
T0*/
_output_shapes
:?????????
*
adj_y(?
 Efficient-CapsNet/digit_caps/mulMul"efficient_capsnet_digit_caps_mul_x:Efficient-CapsNet/digit_caps/digit_cap_attentions:output:0*
T0*/
_output_shapes
:?????????
}
2Efficient-CapsNet/digit_caps/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 Efficient-CapsNet/digit_caps/SumSum$Efficient-CapsNet/digit_caps/mul:z:0;Efficient-CapsNet/digit_caps/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????
*
	keep_dims(c
!Efficient-CapsNet/digit_caps/RankConst*
_output_shapes
: *
dtype0*
value	B :m
"Efficient-CapsNet/digit_caps/add/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 Efficient-CapsNet/digit_caps/addAddV2+Efficient-CapsNet/digit_caps/add/x:output:0*Efficient-CapsNet/digit_caps/Rank:output:0*
T0*
_output_shapes
: e
#Efficient-CapsNet/digit_caps/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :d
"Efficient-CapsNet/digit_caps/mod/yConst*
_output_shapes
: *
dtype0*
value	B :?
 Efficient-CapsNet/digit_caps/modFloorMod$Efficient-CapsNet/digit_caps/add:z:0+Efficient-CapsNet/digit_caps/mod/y:output:0*
T0*
_output_shapes
: d
"Efficient-CapsNet/digit_caps/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :?
 Efficient-CapsNet/digit_caps/SubSub,Efficient-CapsNet/digit_caps/Rank_1:output:0+Efficient-CapsNet/digit_caps/Sub/y:output:0*
T0*
_output_shapes
: j
(Efficient-CapsNet/digit_caps/range/startConst*
_output_shapes
: *
dtype0*
value	B : j
(Efficient-CapsNet/digit_caps/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
"Efficient-CapsNet/digit_caps/rangeRange1Efficient-CapsNet/digit_caps/range/start:output:0$Efficient-CapsNet/digit_caps/mod:z:01Efficient-CapsNet/digit_caps/range/delta:output:0*
_output_shapes
:f
$Efficient-CapsNet/digit_caps/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
"Efficient-CapsNet/digit_caps/add_1AddV2$Efficient-CapsNet/digit_caps/mod:z:0-Efficient-CapsNet/digit_caps/add_1/y:output:0*
T0*
_output_shapes
: l
*Efficient-CapsNet/digit_caps/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
$Efficient-CapsNet/digit_caps/range_1Range&Efficient-CapsNet/digit_caps/add_1:z:0$Efficient-CapsNet/digit_caps/Sub:z:03Efficient-CapsNet/digit_caps/range_1/delta:output:0*
_output_shapes
: ?
,Efficient-CapsNet/digit_caps/concat/values_1Pack$Efficient-CapsNet/digit_caps/Sub:z:0*
N*
T0*
_output_shapes
:?
,Efficient-CapsNet/digit_caps/concat/values_3Pack$Efficient-CapsNet/digit_caps/mod:z:0*
N*
T0*
_output_shapes
:j
(Efficient-CapsNet/digit_caps/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#Efficient-CapsNet/digit_caps/concatConcatV2+Efficient-CapsNet/digit_caps/range:output:05Efficient-CapsNet/digit_caps/concat/values_1:output:0-Efficient-CapsNet/digit_caps/range_1:output:05Efficient-CapsNet/digit_caps/concat/values_3:output:01Efficient-CapsNet/digit_caps/concat/axis:output:0*
N*
T0*
_output_shapes
:?
&Efficient-CapsNet/digit_caps/transpose	Transpose)Efficient-CapsNet/digit_caps/Sum:output:0,Efficient-CapsNet/digit_caps/concat:output:0*
T0*/
_output_shapes
:?????????
?
$Efficient-CapsNet/digit_caps/SoftmaxSoftmax*Efficient-CapsNet/digit_caps/transpose:y:0*
T0*/
_output_shapes
:?????????
f
$Efficient-CapsNet/digit_caps/Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
"Efficient-CapsNet/digit_caps/Sub_1Sub,Efficient-CapsNet/digit_caps/Rank_1:output:0-Efficient-CapsNet/digit_caps/Sub_1/y:output:0*
T0*
_output_shapes
: l
*Efficient-CapsNet/digit_caps/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : l
*Efficient-CapsNet/digit_caps/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
$Efficient-CapsNet/digit_caps/range_2Range3Efficient-CapsNet/digit_caps/range_2/start:output:0$Efficient-CapsNet/digit_caps/mod:z:03Efficient-CapsNet/digit_caps/range_2/delta:output:0*
_output_shapes
:f
$Efficient-CapsNet/digit_caps/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :?
"Efficient-CapsNet/digit_caps/add_2AddV2$Efficient-CapsNet/digit_caps/mod:z:0-Efficient-CapsNet/digit_caps/add_2/y:output:0*
T0*
_output_shapes
: l
*Efficient-CapsNet/digit_caps/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
$Efficient-CapsNet/digit_caps/range_3Range&Efficient-CapsNet/digit_caps/add_2:z:0&Efficient-CapsNet/digit_caps/Sub_1:z:03Efficient-CapsNet/digit_caps/range_3/delta:output:0*
_output_shapes
: ?
.Efficient-CapsNet/digit_caps/concat_1/values_1Pack&Efficient-CapsNet/digit_caps/Sub_1:z:0*
N*
T0*
_output_shapes
:?
.Efficient-CapsNet/digit_caps/concat_1/values_3Pack$Efficient-CapsNet/digit_caps/mod:z:0*
N*
T0*
_output_shapes
:l
*Efficient-CapsNet/digit_caps/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%Efficient-CapsNet/digit_caps/concat_1ConcatV2-Efficient-CapsNet/digit_caps/range_2:output:07Efficient-CapsNet/digit_caps/concat_1/values_1:output:0-Efficient-CapsNet/digit_caps/range_3:output:07Efficient-CapsNet/digit_caps/concat_1/values_3:output:03Efficient-CapsNet/digit_caps/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
<Efficient-CapsNet/digit_caps/digit_cap_coupling_coefficients	Transpose.Efficient-CapsNet/digit_caps/Softmax:softmax:0.Efficient-CapsNet/digit_caps/concat_1:output:0*
T0*/
_output_shapes
:?????????
?
1Efficient-CapsNet/digit_caps/add_3/ReadVariableOpReadVariableOp:efficient_capsnet_digit_caps_add_3_readvariableop_resource*"
_output_shapes
:
*
dtype0?
"Efficient-CapsNet/digit_caps/add_3AddV2@Efficient-CapsNet/digit_caps/digit_cap_coupling_coefficients:y:09Efficient-CapsNet/digit_caps/add_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
?
#Efficient-CapsNet/digit_caps/MatMulBatchMatMulV2&Efficient-CapsNet/digit_caps/add_3:z:0;Efficient-CapsNet/digit_caps/digit_cap_predictions:output:0*
T0*/
_output_shapes
:?????????
?
$Efficient-CapsNet/digit_caps/SqueezeSqueeze,Efficient-CapsNet/digit_caps/MatMul:output:0*
T0*+
_output_shapes
:?????????
*
squeeze_dims

??????????
6Efficient-CapsNet/digit_caps/digit_cap_squash/norm/mulMul-Efficient-CapsNet/digit_caps/Squeeze:output:0-Efficient-CapsNet/digit_caps/Squeeze:output:0*
T0*+
_output_shapes
:?????????
?
HEfficient-CapsNet/digit_caps/digit_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
6Efficient-CapsNet/digit_caps/digit_cap_squash/norm/SumSum:Efficient-CapsNet/digit_caps/digit_cap_squash/norm/mul:z:0QEfficient-CapsNet/digit_caps/digit_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????
*
	keep_dims(?
7Efficient-CapsNet/digit_caps/digit_cap_squash/norm/SqrtSqrt?Efficient-CapsNet/digit_caps/digit_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:?????????
?
1Efficient-CapsNet/digit_caps/digit_cap_squash/ExpExp;Efficient-CapsNet/digit_caps/digit_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:?????????
|
7Efficient-CapsNet/digit_caps/digit_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
5Efficient-CapsNet/digit_caps/digit_cap_squash/truedivRealDiv@Efficient-CapsNet/digit_caps/digit_cap_squash/truediv/x:output:05Efficient-CapsNet/digit_caps/digit_cap_squash/Exp:y:0*
T0*+
_output_shapes
:?????????
x
3Efficient-CapsNet/digit_caps/digit_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
1Efficient-CapsNet/digit_caps/digit_cap_squash/subSub<Efficient-CapsNet/digit_caps/digit_cap_squash/sub/x:output:09Efficient-CapsNet/digit_caps/digit_cap_squash/truediv:z:0*
T0*+
_output_shapes
:?????????
x
3Efficient-CapsNet/digit_caps/digit_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
1Efficient-CapsNet/digit_caps/digit_cap_squash/addAddV2;Efficient-CapsNet/digit_caps/digit_cap_squash/norm/Sqrt:y:0<Efficient-CapsNet/digit_caps/digit_cap_squash/add/y:output:0*
T0*+
_output_shapes
:?????????
?
7Efficient-CapsNet/digit_caps/digit_cap_squash/truediv_1RealDiv-Efficient-CapsNet/digit_caps/Squeeze:output:05Efficient-CapsNet/digit_caps/digit_cap_squash/add:z:0*
T0*+
_output_shapes
:?????????
?
1Efficient-CapsNet/digit_caps/digit_cap_squash/mulMul5Efficient-CapsNet/digit_caps/digit_cap_squash/sub:z:0;Efficient-CapsNet/digit_caps/digit_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:?????????
?
&Efficient-CapsNet/digit_probs/norm/mulMul5Efficient-CapsNet/digit_caps/digit_cap_squash/mul:z:05Efficient-CapsNet/digit_caps/digit_cap_squash/mul:z:0*
T0*+
_output_shapes
:?????????
?
8Efficient-CapsNet/digit_probs/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
&Efficient-CapsNet/digit_probs/norm/SumSum*Efficient-CapsNet/digit_probs/norm/mul:z:0AEfficient-CapsNet/digit_probs/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????
*
	keep_dims(?
'Efficient-CapsNet/digit_probs/norm/SqrtSqrt/Efficient-CapsNet/digit_probs/norm/Sum:output:0*
T0*+
_output_shapes
:?????????
?
*Efficient-CapsNet/digit_probs/norm/SqueezeSqueeze+Efficient-CapsNet/digit_probs/norm/Sqrt:y:0*
T0*'
_output_shapes
:?????????
*
squeeze_dims

??????????
IdentityIdentity3Efficient-CapsNet/digit_probs/norm/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp2^Efficient-CapsNet/digit_caps/add_3/ReadVariableOp'^Efficient-CapsNet/digit_caps/map/whileH^Efficient-CapsNet/feature_maps/feature_map_conv1/BiasAdd/ReadVariableOpG^Efficient-CapsNet/feature_maps/feature_map_conv1/Conv2D/ReadVariableOpH^Efficient-CapsNet/feature_maps/feature_map_conv2/BiasAdd/ReadVariableOpG^Efficient-CapsNet/feature_maps/feature_map_conv2/Conv2D/ReadVariableOpH^Efficient-CapsNet/feature_maps/feature_map_conv3/BiasAdd/ReadVariableOpG^Efficient-CapsNet/feature_maps/feature_map_conv3/Conv2D/ReadVariableOpH^Efficient-CapsNet/feature_maps/feature_map_conv4/BiasAdd/ReadVariableOpG^Efficient-CapsNet/feature_maps/feature_map_conv4/Conv2D/ReadVariableOpQ^Efficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOpS^Efficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1@^Efficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOpB^Efficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOp_1Q^Efficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOpS^Efficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1@^Efficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOpB^Efficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOp_1Q^Efficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOpS^Efficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1@^Efficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOpB^Efficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOp_1Q^Efficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOpS^Efficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1@^Efficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOpB^Efficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOp_1H^Efficient-CapsNet/primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOpG^Efficient-CapsNet/primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2f
1Efficient-CapsNet/digit_caps/add_3/ReadVariableOp1Efficient-CapsNet/digit_caps/add_3/ReadVariableOp2P
&Efficient-CapsNet/digit_caps/map/while&Efficient-CapsNet/digit_caps/map/while2?
GEfficient-CapsNet/feature_maps/feature_map_conv1/BiasAdd/ReadVariableOpGEfficient-CapsNet/feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp2?
FEfficient-CapsNet/feature_maps/feature_map_conv1/Conv2D/ReadVariableOpFEfficient-CapsNet/feature_maps/feature_map_conv1/Conv2D/ReadVariableOp2?
GEfficient-CapsNet/feature_maps/feature_map_conv2/BiasAdd/ReadVariableOpGEfficient-CapsNet/feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp2?
FEfficient-CapsNet/feature_maps/feature_map_conv2/Conv2D/ReadVariableOpFEfficient-CapsNet/feature_maps/feature_map_conv2/Conv2D/ReadVariableOp2?
GEfficient-CapsNet/feature_maps/feature_map_conv3/BiasAdd/ReadVariableOpGEfficient-CapsNet/feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp2?
FEfficient-CapsNet/feature_maps/feature_map_conv3/Conv2D/ReadVariableOpFEfficient-CapsNet/feature_maps/feature_map_conv3/Conv2D/ReadVariableOp2?
GEfficient-CapsNet/feature_maps/feature_map_conv4/BiasAdd/ReadVariableOpGEfficient-CapsNet/feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp2?
FEfficient-CapsNet/feature_maps/feature_map_conv4/Conv2D/ReadVariableOpFEfficient-CapsNet/feature_maps/feature_map_conv4/Conv2D/ReadVariableOp2?
PEfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOpPEfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp2?
REfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1REfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_12?
?Efficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOp?Efficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOp2?
AEfficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOp_1AEfficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOp_12?
PEfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOpPEfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp2?
REfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1REfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_12?
?Efficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOp?Efficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOp2?
AEfficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOp_1AEfficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOp_12?
PEfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOpPEfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp2?
REfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1REfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_12?
?Efficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOp?Efficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOp2?
AEfficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOp_1AEfficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOp_12?
PEfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOpPEfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp2?
REfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1REfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_12?
?Efficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOp?Efficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOp2?
AEfficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOp_1AEfficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOp_12?
GEfficient-CapsNet/primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOpGEfficient-CapsNet/primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp2?
FEfficient-CapsNet/primary_caps/primary_cap_dconv/Conv2D/ReadVariableOpFEfficient-CapsNet/primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:?????????
&
_user_specified_nameinput_images:

_output_shapes
: 
?
?
$__inference_signature_wrapper_726349
input_images!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?%

unknown_23:		?

unknown_24:	?$

unknown_25:


unknown_26 

unknown_27:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_imagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_724390o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameinput_images:

_output_shapes
: 
?
?
2__inference_Efficient-CapsNet_layer_call_fn_725794

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?%

unknown_23:		?

unknown_24:	?$

unknown_25:


unknown_26 

unknown_27:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_725406o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
 digit_caps_map_while_cond_725928:
6digit_caps_map_while_digit_caps_map_while_loop_counter5
1digit_caps_map_while_digit_caps_map_strided_slice$
 digit_caps_map_while_placeholder&
"digit_caps_map_while_placeholder_1:
6digit_caps_map_while_less_digit_caps_map_strided_sliceR
Ndigit_caps_map_while_digit_caps_map_while_cond_725928___redundant_placeholder0R
Ndigit_caps_map_while_digit_caps_map_while_cond_725928___redundant_placeholder1!
digit_caps_map_while_identity
?
digit_caps/map/while/LessLess digit_caps_map_while_placeholder6digit_caps_map_while_less_digit_caps_map_strided_slice*
T0*
_output_shapes
: ?
digit_caps/map/while/Less_1Less6digit_caps_map_while_digit_caps_map_while_loop_counter1digit_caps_map_while_digit_caps_map_strided_slice*
T0*
_output_shapes
: ?
digit_caps/map/while/LogicalAnd
LogicalAnddigit_caps/map/while/Less_1:z:0digit_caps/map/while/Less:z:0*
_output_shapes
: o
digit_caps/map/while/IdentityIdentity#digit_caps/map/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "G
digit_caps_map_while_identity&digit_caps/map/while/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
?
?
2__inference_Efficient-CapsNet_layer_call_fn_725530
input_images!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?%

unknown_23:		?

unknown_24:	?$

unknown_25:


unknown_26 

unknown_27:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_imagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_725406o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameinput_images:

_output_shapes
: 
?O
?
F__inference_digit_caps_layer_call_and_return_conditional_losses_726811
primary_caps+
map_while_input_6:
	
mul_x3
add_3_readvariableop_resource:

identity??add_3/ReadVariableOp?	map/whileP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :y

ExpandDims
ExpandDimsprimary_capsExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????g
Tile/multiplesConst*
_output_shapes
:*
dtype0*%
valueB"   
         t
TileTileExpandDims:output:0Tile/multiples:output:0*
T0*/
_output_shapes
:?????????
_
digit_cap_inputs/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
digit_cap_inputs
ExpandDimsTile:output:0digit_cap_inputs/dim:output:0*
T0*3
_output_shapes!
:?????????
R
	map/ShapeShapedigit_cap_inputs:output:0*
T0*
_output_shapes
:a
map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
map/strided_sliceStridedSlicemap/Shape:output:0 map/strided_slice/stack:output:0"map/strided_slice/stack_1:output:0"map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            ?
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensordigit_cap_inputs:output:0Bmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???K
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : l
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???X
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
	map/whileWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_1:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0map_while_input_6*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : *#
_read_only_resource_inputs
*
_stateful_parallelism( *!
bodyR
map_while_body_726706*!
condR
map_while_cond_726705*!
output_shapes
: : : : : : : ?
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            ?
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:?????????
*
element_dtype0?
digit_cap_predictionsSqueeze/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*/
_output_shapes
:?????????
*
squeeze_dims

??????????
digit_cap_attentionsBatchMatMulV2digit_cap_predictions:output:0digit_cap_predictions:output:0*
T0*/
_output_shapes
:?????????
*
adj_y(j
mulMulmul_xdigit_cap_attentions:output:0*
T0*/
_output_shapes
:?????????
`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????~
SumSummul:z:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????
*
	keep_dims(F
RankConst*
_output_shapes
: *
dtype0*
value	B :P
add/xConst*
_output_shapes
: *
dtype0*
valueB :
?????????L
addAddV2add/x:output:0Rank:output:0*
T0*
_output_shapes
: H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :G
mod/yConst*
_output_shapes
: *
dtype0*
value	B :I
modFloorModadd:z:0mod/y:output:0*
T0*
_output_shapes
: G
Sub/yConst*
_output_shapes
: *
dtype0*
value	B :L
SubSubRank_1:output:0Sub/y:output:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :_
rangeRangerange/start:output:0mod:z:0range/delta:output:0*
_output_shapes
:I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :J
add_1AddV2mod:z:0add_1/y:output:0*
T0*
_output_shapes
: O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :V
range_1Range	add_1:z:0Sub:z:0range_1/delta:output:0*
_output_shapes
: N
concat/values_1PackSub:z:0*
N*
T0*
_output_shapes
:N
concat/values_3Packmod:z:0*
N*
T0*
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2range:output:0concat/values_1:output:0range_1:output:0concat/values_3:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:o
	transpose	TransposeSum:output:0concat:output:0*
T0*/
_output_shapes
:?????????
[
SoftmaxSoftmaxtranspose:y:0*
T0*/
_output_shapes
:?????????
I
Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :P
Sub_1SubRank_1:output:0Sub_1/y:output:0*
T0*
_output_shapes
: O
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
range_2Rangerange_2/start:output:0mod:z:0range_2/delta:output:0*
_output_shapes
:I
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :J
add_2AddV2mod:z:0add_2/y:output:0*
T0*
_output_shapes
: O
range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :X
range_3Range	add_2:z:0	Sub_1:z:0range_3/delta:output:0*
_output_shapes
: R
concat_1/values_1Pack	Sub_1:z:0*
N*
T0*
_output_shapes
:P
concat_1/values_3Packmod:z:0*
N*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_1ConcatV2range_2:output:0concat_1/values_1:output:0range_3:output:0concat_1/values_3:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
digit_cap_coupling_coefficients	TransposeSoftmax:softmax:0concat_1:output:0*
T0*/
_output_shapes
:?????????
v
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*"
_output_shapes
:
*
dtype0?
add_3AddV2#digit_cap_coupling_coefficients:y:0add_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
|
MatMulBatchMatMulV2	add_3:z:0digit_cap_predictions:output:0*
T0*/
_output_shapes
:?????????
y
SqueezeSqueezeMatMul:output:0*
T0*+
_output_shapes
:?????????
*
squeeze_dims

?????????z
digit_cap_squash/norm/mulMulSqueeze:output:0Squeeze:output:0*
T0*+
_output_shapes
:?????????
~
+digit_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
digit_cap_squash/norm/SumSumdigit_cap_squash/norm/mul:z:04digit_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????
*
	keep_dims(|
digit_cap_squash/norm/SqrtSqrt"digit_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:?????????
q
digit_cap_squash/ExpExpdigit_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:?????????
_
digit_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
digit_cap_squash/truedivRealDiv#digit_cap_squash/truediv/x:output:0digit_cap_squash/Exp:y:0*
T0*+
_output_shapes
:?????????
[
digit_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
digit_cap_squash/subSubdigit_cap_squash/sub/x:output:0digit_cap_squash/truediv:z:0*
T0*+
_output_shapes
:?????????
[
digit_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
digit_cap_squash/addAddV2digit_cap_squash/norm/Sqrt:y:0digit_cap_squash/add/y:output:0*
T0*+
_output_shapes
:?????????
?
digit_cap_squash/truediv_1RealDivSqueeze:output:0digit_cap_squash/add:z:0*
T0*+
_output_shapes
:?????????
?
digit_cap_squash/mulMuldigit_cap_squash/sub:z:0digit_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:?????????
k
IdentityIdentitydigit_cap_squash/mul:z:0^NoOp*
T0*+
_output_shapes
:?????????
i
NoOpNoOp^add_3/ReadVariableOp
^map/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2,
add_3/ReadVariableOpadd_3/ReadVariableOp2
	map/while	map/while:Y U
+
_output_shapes
:?????????
&
_user_specified_nameprimary_caps:

_output_shapes
: 
?
?
M__inference_feature_map_norm3_layer_call_and_return_conditional_losses_727025

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
 digit_caps_map_while_cond_726173:
6digit_caps_map_while_digit_caps_map_while_loop_counter5
1digit_caps_map_while_digit_caps_map_strided_slice$
 digit_caps_map_while_placeholder&
"digit_caps_map_while_placeholder_1:
6digit_caps_map_while_less_digit_caps_map_strided_sliceR
Ndigit_caps_map_while_digit_caps_map_while_cond_726173___redundant_placeholder0R
Ndigit_caps_map_while_digit_caps_map_while_cond_726173___redundant_placeholder1!
digit_caps_map_while_identity
?
digit_caps/map/while/LessLess digit_caps_map_while_placeholder6digit_caps_map_while_less_digit_caps_map_strided_slice*
T0*
_output_shapes
: ?
digit_caps/map/while/Less_1Less6digit_caps_map_while_digit_caps_map_while_loop_counter1digit_caps_map_while_digit_caps_map_strided_slice*
T0*
_output_shapes
: ?
digit_caps/map/while/LogicalAnd
LogicalAnddigit_caps/map/while/Less_1:z:0digit_caps/map/while/Less:z:0*
_output_shapes
: o
digit_caps/map/while/IdentityIdentity#digit_caps/map/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "G
digit_caps_map_while_identity&digit_caps/map/while/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
?
?
M__inference_feature_map_norm1_layer_call_and_return_conditional_losses_726883

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
-__inference_feature_maps_layer_call_fn_726402
input_images!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_imagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_feature_maps_layer_call_and_return_conditional_losses_724741x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????		?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameinput_images
?
?
-__inference_feature_maps_layer_call_fn_726455
input_images!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_imagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_feature_maps_layer_call_and_return_conditional_losses_725224x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????		?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameinput_images
?
c
G__inference_digit_probs_layer_call_and_return_conditional_losses_726839

inputs
identityU
norm/mulMulinputsinputs*
T0*+
_output_shapes
:?????????
m
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????
*
	keep_dims(Z
	norm/SqrtSqrtnorm/Sum:output:0*
T0*+
_output_shapes
:?????????
x
norm/SqueezeSqueezenorm/Sqrt:y:0*
T0*'
_output_shapes
:?????????
*
squeeze_dims

?????????]
IdentityIdentitynorm/Squeeze:output:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?!
?

M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_725596
input_images-
feature_maps_725533: !
feature_maps_725535: !
feature_maps_725537: !
feature_maps_725539: !
feature_maps_725541: !
feature_maps_725543: -
feature_maps_725545: @!
feature_maps_725547:@!
feature_maps_725549:@!
feature_maps_725551:@!
feature_maps_725553:@!
feature_maps_725555:@-
feature_maps_725557:@@!
feature_maps_725559:@!
feature_maps_725561:@!
feature_maps_725563:@!
feature_maps_725565:@!
feature_maps_725567:@.
feature_maps_725569:@?"
feature_maps_725571:	?"
feature_maps_725573:	?"
feature_maps_725575:	?"
feature_maps_725577:	?"
feature_maps_725579:	?.
primary_caps_725582:		?"
primary_caps_725584:	?+
digit_caps_725587:

digit_caps_725589'
digit_caps_725591:

identity??"digit_caps/StatefulPartitionedCall?$feature_maps/StatefulPartitionedCall?$primary_caps/StatefulPartitionedCall?
$feature_maps/StatefulPartitionedCallStatefulPartitionedCallinput_imagesfeature_maps_725533feature_maps_725535feature_maps_725537feature_maps_725539feature_maps_725541feature_maps_725543feature_maps_725545feature_maps_725547feature_maps_725549feature_maps_725551feature_maps_725553feature_maps_725555feature_maps_725557feature_maps_725559feature_maps_725561feature_maps_725563feature_maps_725565feature_maps_725567feature_maps_725569feature_maps_725571feature_maps_725573feature_maps_725575feature_maps_725577feature_maps_725579*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_feature_maps_layer_call_and_return_conditional_losses_724741?
$primary_caps/StatefulPartitionedCallStatefulPartitionedCall-feature_maps/StatefulPartitionedCall:output:0primary_caps_725582primary_caps_725584*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_primary_caps_layer_call_and_return_conditional_losses_724824?
"digit_caps/StatefulPartitionedCallStatefulPartitionedCall-primary_caps/StatefulPartitionedCall:output:0digit_caps_725587digit_caps_725589digit_caps_725591*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_digit_caps_layer_call_and_return_conditional_losses_724957?
digit_probs/PartitionedCallPartitionedCall+digit_caps/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_digit_probs_layer_call_and_return_conditional_losses_724974s
IdentityIdentity$digit_probs/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp#^digit_caps/StatefulPartitionedCall%^feature_maps/StatefulPartitionedCall%^primary_caps/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"digit_caps/StatefulPartitionedCall"digit_caps/StatefulPartitionedCall2L
$feature_maps/StatefulPartitionedCall$feature_maps/StatefulPartitionedCall2L
$primary_caps/StatefulPartitionedCall$primary_caps/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameinput_images:

_output_shapes
: 
?
?
2__inference_Efficient-CapsNet_layer_call_fn_725731

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?%

unknown_23:		?

unknown_24:	?$

unknown_25:


unknown_26 

unknown_27:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_724977o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
M__inference_feature_map_norm4_layer_call_and_return_conditional_losses_727069

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?!
?

M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_725662
input_images-
feature_maps_725599: !
feature_maps_725601: !
feature_maps_725603: !
feature_maps_725605: !
feature_maps_725607: !
feature_maps_725609: -
feature_maps_725611: @!
feature_maps_725613:@!
feature_maps_725615:@!
feature_maps_725617:@!
feature_maps_725619:@!
feature_maps_725621:@-
feature_maps_725623:@@!
feature_maps_725625:@!
feature_maps_725627:@!
feature_maps_725629:@!
feature_maps_725631:@!
feature_maps_725633:@.
feature_maps_725635:@?"
feature_maps_725637:	?"
feature_maps_725639:	?"
feature_maps_725641:	?"
feature_maps_725643:	?"
feature_maps_725645:	?.
primary_caps_725648:		?"
primary_caps_725650:	?+
digit_caps_725653:

digit_caps_725655'
digit_caps_725657:

identity??"digit_caps/StatefulPartitionedCall?$feature_maps/StatefulPartitionedCall?$primary_caps/StatefulPartitionedCall?
$feature_maps/StatefulPartitionedCallStatefulPartitionedCallinput_imagesfeature_maps_725599feature_maps_725601feature_maps_725603feature_maps_725605feature_maps_725607feature_maps_725609feature_maps_725611feature_maps_725613feature_maps_725615feature_maps_725617feature_maps_725619feature_maps_725621feature_maps_725623feature_maps_725625feature_maps_725627feature_maps_725629feature_maps_725631feature_maps_725633feature_maps_725635feature_maps_725637feature_maps_725639feature_maps_725641feature_maps_725643feature_maps_725645*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_feature_maps_layer_call_and_return_conditional_losses_725224?
$primary_caps/StatefulPartitionedCallStatefulPartitionedCall-feature_maps/StatefulPartitionedCall:output:0primary_caps_725648primary_caps_725650*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_primary_caps_layer_call_and_return_conditional_losses_724824?
"digit_caps/StatefulPartitionedCallStatefulPartitionedCall-primary_caps/StatefulPartitionedCall:output:0digit_caps_725653digit_caps_725655digit_caps_725657*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_digit_caps_layer_call_and_return_conditional_losses_724957?
digit_probs/PartitionedCallPartitionedCall+digit_caps/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_digit_probs_layer_call_and_return_conditional_losses_725055s
IdentityIdentity$digit_probs/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp#^digit_caps/StatefulPartitionedCall%^feature_maps/StatefulPartitionedCall%^primary_caps/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"digit_caps/StatefulPartitionedCall"digit_caps/StatefulPartitionedCall2L
$feature_maps/StatefulPartitionedCall$feature_maps/StatefulPartitionedCall2L
$primary_caps/StatefulPartitionedCall$primary_caps/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameinput_images:

_output_shapes
: 
??
?
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_726039

inputsW
=feature_maps_feature_map_conv1_conv2d_readvariableop_resource: L
>feature_maps_feature_map_conv1_biasadd_readvariableop_resource: D
6feature_maps_feature_map_norm1_readvariableop_resource: F
8feature_maps_feature_map_norm1_readvariableop_1_resource: U
Gfeature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_resource: W
Ifeature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource: W
=feature_maps_feature_map_conv2_conv2d_readvariableop_resource: @L
>feature_maps_feature_map_conv2_biasadd_readvariableop_resource:@D
6feature_maps_feature_map_norm2_readvariableop_resource:@F
8feature_maps_feature_map_norm2_readvariableop_1_resource:@U
Gfeature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_resource:@W
Ifeature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource:@W
=feature_maps_feature_map_conv3_conv2d_readvariableop_resource:@@L
>feature_maps_feature_map_conv3_biasadd_readvariableop_resource:@D
6feature_maps_feature_map_norm3_readvariableop_resource:@F
8feature_maps_feature_map_norm3_readvariableop_1_resource:@U
Gfeature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_resource:@W
Ifeature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource:@X
=feature_maps_feature_map_conv4_conv2d_readvariableop_resource:@?M
>feature_maps_feature_map_conv4_biasadd_readvariableop_resource:	?E
6feature_maps_feature_map_norm4_readvariableop_resource:	?G
8feature_maps_feature_map_norm4_readvariableop_1_resource:	?V
Gfeature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_resource:	?X
Ifeature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource:	?X
=primary_caps_primary_cap_dconv_conv2d_readvariableop_resource:		?M
>primary_caps_primary_cap_dconv_biasadd_readvariableop_resource:	?6
digit_caps_map_while_input_6:

digit_caps_mul_x>
(digit_caps_add_3_readvariableop_resource:

identity??digit_caps/add_3/ReadVariableOp?digit_caps/map/while?5feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp?4feature_maps/feature_map_conv1/Conv2D/ReadVariableOp?5feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp?4feature_maps/feature_map_conv2/Conv2D/ReadVariableOp?5feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp?4feature_maps/feature_map_conv3/Conv2D/ReadVariableOp?5feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp?4feature_maps/feature_map_conv4/Conv2D/ReadVariableOp?>feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp?@feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1?-feature_maps/feature_map_norm1/ReadVariableOp?/feature_maps/feature_map_norm1/ReadVariableOp_1?>feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp?@feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1?-feature_maps/feature_map_norm2/ReadVariableOp?/feature_maps/feature_map_norm2/ReadVariableOp_1?>feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp?@feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1?-feature_maps/feature_map_norm3/ReadVariableOp?/feature_maps/feature_map_norm3/ReadVariableOp_1?>feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp?@feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1?-feature_maps/feature_map_norm4/ReadVariableOp?/feature_maps/feature_map_norm4/ReadVariableOp_1?5primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp?4primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp?
4feature_maps/feature_map_conv1/Conv2D/ReadVariableOpReadVariableOp=feature_maps_feature_map_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
%feature_maps/feature_map_conv1/Conv2DConv2Dinputs<feature_maps/feature_map_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
5feature_maps/feature_map_conv1/BiasAdd/ReadVariableOpReadVariableOp>feature_maps_feature_map_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
&feature_maps/feature_map_conv1/BiasAddBiasAdd.feature_maps/feature_map_conv1/Conv2D:output:0=feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
#feature_maps/feature_map_conv1/ReluRelu/feature_maps/feature_map_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
-feature_maps/feature_map_norm1/ReadVariableOpReadVariableOp6feature_maps_feature_map_norm1_readvariableop_resource*
_output_shapes
: *
dtype0?
/feature_maps/feature_map_norm1/ReadVariableOp_1ReadVariableOp8feature_maps_feature_map_norm1_readvariableop_1_resource*
_output_shapes
: *
dtype0?
>feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOpReadVariableOpGfeature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
@feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIfeature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
/feature_maps/feature_map_norm1/FusedBatchNormV3FusedBatchNormV31feature_maps/feature_map_conv1/Relu:activations:05feature_maps/feature_map_norm1/ReadVariableOp:value:07feature_maps/feature_map_norm1/ReadVariableOp_1:value:0Ffeature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp:value:0Hfeature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
4feature_maps/feature_map_conv2/Conv2D/ReadVariableOpReadVariableOp=feature_maps_feature_map_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
%feature_maps/feature_map_conv2/Conv2DConv2D3feature_maps/feature_map_norm1/FusedBatchNormV3:y:0<feature_maps/feature_map_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
5feature_maps/feature_map_conv2/BiasAdd/ReadVariableOpReadVariableOp>feature_maps_feature_map_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
&feature_maps/feature_map_conv2/BiasAddBiasAdd.feature_maps/feature_map_conv2/Conv2D:output:0=feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
#feature_maps/feature_map_conv2/ReluRelu/feature_maps/feature_map_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
-feature_maps/feature_map_norm2/ReadVariableOpReadVariableOp6feature_maps_feature_map_norm2_readvariableop_resource*
_output_shapes
:@*
dtype0?
/feature_maps/feature_map_norm2/ReadVariableOp_1ReadVariableOp8feature_maps_feature_map_norm2_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
>feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOpReadVariableOpGfeature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
@feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIfeature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
/feature_maps/feature_map_norm2/FusedBatchNormV3FusedBatchNormV31feature_maps/feature_map_conv2/Relu:activations:05feature_maps/feature_map_norm2/ReadVariableOp:value:07feature_maps/feature_map_norm2/ReadVariableOp_1:value:0Ffeature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp:value:0Hfeature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
4feature_maps/feature_map_conv3/Conv2D/ReadVariableOpReadVariableOp=feature_maps_feature_map_conv3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
%feature_maps/feature_map_conv3/Conv2DConv2D3feature_maps/feature_map_norm2/FusedBatchNormV3:y:0<feature_maps/feature_map_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
5feature_maps/feature_map_conv3/BiasAdd/ReadVariableOpReadVariableOp>feature_maps_feature_map_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
&feature_maps/feature_map_conv3/BiasAddBiasAdd.feature_maps/feature_map_conv3/Conv2D:output:0=feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
#feature_maps/feature_map_conv3/ReluRelu/feature_maps/feature_map_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
-feature_maps/feature_map_norm3/ReadVariableOpReadVariableOp6feature_maps_feature_map_norm3_readvariableop_resource*
_output_shapes
:@*
dtype0?
/feature_maps/feature_map_norm3/ReadVariableOp_1ReadVariableOp8feature_maps_feature_map_norm3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
>feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOpReadVariableOpGfeature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
@feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIfeature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
/feature_maps/feature_map_norm3/FusedBatchNormV3FusedBatchNormV31feature_maps/feature_map_conv3/Relu:activations:05feature_maps/feature_map_norm3/ReadVariableOp:value:07feature_maps/feature_map_norm3/ReadVariableOp_1:value:0Ffeature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp:value:0Hfeature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
4feature_maps/feature_map_conv4/Conv2D/ReadVariableOpReadVariableOp=feature_maps_feature_map_conv4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
%feature_maps/feature_map_conv4/Conv2DConv2D3feature_maps/feature_map_norm3/FusedBatchNormV3:y:0<feature_maps/feature_map_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
?
5feature_maps/feature_map_conv4/BiasAdd/ReadVariableOpReadVariableOp>feature_maps_feature_map_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&feature_maps/feature_map_conv4/BiasAddBiasAdd.feature_maps/feature_map_conv4/Conv2D:output:0=feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		??
#feature_maps/feature_map_conv4/ReluRelu/feature_maps/feature_map_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		??
-feature_maps/feature_map_norm4/ReadVariableOpReadVariableOp6feature_maps_feature_map_norm4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
/feature_maps/feature_map_norm4/ReadVariableOp_1ReadVariableOp8feature_maps_feature_map_norm4_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
>feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOpReadVariableOpGfeature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
@feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIfeature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
/feature_maps/feature_map_norm4/FusedBatchNormV3FusedBatchNormV31feature_maps/feature_map_conv4/Relu:activations:05feature_maps/feature_map_norm4/ReadVariableOp:value:07feature_maps/feature_map_norm4/ReadVariableOp_1:value:0Ffeature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp:value:0Hfeature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????		?:?:?:?:?:*
epsilon%o?:*
is_training( ?
4primary_caps/primary_cap_dconv/Conv2D/ReadVariableOpReadVariableOp=primary_caps_primary_cap_dconv_conv2d_readvariableop_resource*'
_output_shapes
:		?*
dtype0?
%primary_caps/primary_cap_dconv/Conv2DConv2D3feature_maps/feature_map_norm4/FusedBatchNormV3:y:0<primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
5primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOpReadVariableOp>primary_caps_primary_cap_dconv_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&primary_caps/primary_cap_dconv/BiasAddBiasAdd.primary_caps/primary_cap_dconv/Conv2D:output:0=primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
#primary_caps/primary_cap_dconv/ReluRelu/primary_caps/primary_cap_dconv/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
&primary_caps/primary_cap_reshape/ShapeShape1primary_caps/primary_cap_dconv/Relu:activations:0*
T0*
_output_shapes
:~
4primary_caps/primary_cap_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6primary_caps/primary_cap_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6primary_caps/primary_cap_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.primary_caps/primary_cap_reshape/strided_sliceStridedSlice/primary_caps/primary_cap_reshape/Shape:output:0=primary_caps/primary_cap_reshape/strided_slice/stack:output:0?primary_caps/primary_cap_reshape/strided_slice/stack_1:output:0?primary_caps/primary_cap_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0primary_caps/primary_cap_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????r
0primary_caps/primary_cap_reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
.primary_caps/primary_cap_reshape/Reshape/shapePack7primary_caps/primary_cap_reshape/strided_slice:output:09primary_caps/primary_cap_reshape/Reshape/shape/1:output:09primary_caps/primary_cap_reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
(primary_caps/primary_cap_reshape/ReshapeReshape1primary_caps/primary_cap_dconv/Relu:activations:07primary_caps/primary_cap_reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:??????????
(primary_caps/primary_cap_squash/norm/mulMul1primary_caps/primary_cap_reshape/Reshape:output:01primary_caps/primary_cap_reshape/Reshape:output:0*
T0*+
_output_shapes
:??????????
:primary_caps/primary_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
(primary_caps/primary_cap_squash/norm/SumSum,primary_caps/primary_cap_squash/norm/mul:z:0Cprimary_caps/primary_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(?
)primary_caps/primary_cap_squash/norm/SqrtSqrt1primary_caps/primary_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:??????????
#primary_caps/primary_cap_squash/ExpExp-primary_caps/primary_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:?????????n
)primary_caps/primary_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'primary_caps/primary_cap_squash/truedivRealDiv2primary_caps/primary_cap_squash/truediv/x:output:0'primary_caps/primary_cap_squash/Exp:y:0*
T0*+
_output_shapes
:?????????j
%primary_caps/primary_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#primary_caps/primary_cap_squash/subSub.primary_caps/primary_cap_squash/sub/x:output:0+primary_caps/primary_cap_squash/truediv:z:0*
T0*+
_output_shapes
:?????????j
%primary_caps/primary_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
#primary_caps/primary_cap_squash/addAddV2-primary_caps/primary_cap_squash/norm/Sqrt:y:0.primary_caps/primary_cap_squash/add/y:output:0*
T0*+
_output_shapes
:??????????
)primary_caps/primary_cap_squash/truediv_1RealDiv1primary_caps/primary_cap_reshape/Reshape:output:0'primary_caps/primary_cap_squash/add:z:0*
T0*+
_output_shapes
:??????????
#primary_caps/primary_cap_squash/mulMul'primary_caps/primary_cap_squash/sub:z:0-primary_caps/primary_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:?????????[
digit_caps/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
digit_caps/ExpandDims
ExpandDims'primary_caps/primary_cap_squash/mul:z:0"digit_caps/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????r
digit_caps/Tile/multiplesConst*
_output_shapes
:*
dtype0*%
valueB"   
         ?
digit_caps/TileTiledigit_caps/ExpandDims:output:0"digit_caps/Tile/multiples:output:0*
T0*/
_output_shapes
:?????????
j
digit_caps/digit_cap_inputs/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
digit_caps/digit_cap_inputs
ExpandDimsdigit_caps/Tile:output:0(digit_caps/digit_cap_inputs/dim:output:0*
T0*3
_output_shapes!
:?????????
h
digit_caps/map/ShapeShape$digit_caps/digit_cap_inputs:output:0*
T0*
_output_shapes
:l
"digit_caps/map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$digit_caps/map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$digit_caps/map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
digit_caps/map/strided_sliceStridedSlicedigit_caps/map/Shape:output:0+digit_caps/map/strided_slice/stack:output:0-digit_caps/map/strided_slice/stack_1:output:0-digit_caps/map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
*digit_caps/map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
digit_caps/map/TensorArrayV2TensorListReserve3digit_caps/map/TensorArrayV2/element_shape:output:0%digit_caps/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Ddigit_caps/map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            ?
6digit_caps/map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$digit_caps/digit_cap_inputs:output:0Mdigit_caps/map/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???V
digit_caps/map/ConstConst*
_output_shapes
: *
dtype0*
value	B : w
,digit_caps/map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
digit_caps/map/TensorArrayV2_1TensorListReserve5digit_caps/map/TensorArrayV2_1/element_shape:output:0%digit_caps/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???c
!digit_caps/map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
digit_caps/map/whileWhile*digit_caps/map/while/loop_counter:output:0%digit_caps/map/strided_slice:output:0digit_caps/map/Const:output:0'digit_caps/map/TensorArrayV2_1:handle:0%digit_caps/map/strided_slice:output:0Fdigit_caps/map/TensorArrayUnstack/TensorListFromTensor:output_handle:0digit_caps_map_while_input_6*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : *#
_read_only_resource_inputs
*
_stateful_parallelism( *,
body$R"
 digit_caps_map_while_body_725929*,
cond$R"
 digit_caps_map_while_cond_725928*!
output_shapes
: : : : : : : ?
?digit_caps/map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            ?
1digit_caps/map/TensorArrayV2Stack/TensorListStackTensorListStackdigit_caps/map/while:output:3Hdigit_caps/map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:?????????
*
element_dtype0?
 digit_caps/digit_cap_predictionsSqueeze:digit_caps/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*/
_output_shapes
:?????????
*
squeeze_dims

??????????
digit_caps/digit_cap_attentionsBatchMatMulV2)digit_caps/digit_cap_predictions:output:0)digit_caps/digit_cap_predictions:output:0*
T0*/
_output_shapes
:?????????
*
adj_y(?
digit_caps/mulMuldigit_caps_mul_x(digit_caps/digit_cap_attentions:output:0*
T0*/
_output_shapes
:?????????
k
 digit_caps/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
digit_caps/SumSumdigit_caps/mul:z:0)digit_caps/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????
*
	keep_dims(Q
digit_caps/RankConst*
_output_shapes
: *
dtype0*
value	B :[
digit_caps/add/xConst*
_output_shapes
: *
dtype0*
valueB :
?????????m
digit_caps/addAddV2digit_caps/add/x:output:0digit_caps/Rank:output:0*
T0*
_output_shapes
: S
digit_caps/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :R
digit_caps/mod/yConst*
_output_shapes
: *
dtype0*
value	B :j
digit_caps/modFloorModdigit_caps/add:z:0digit_caps/mod/y:output:0*
T0*
_output_shapes
: R
digit_caps/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :m
digit_caps/SubSubdigit_caps/Rank_1:output:0digit_caps/Sub/y:output:0*
T0*
_output_shapes
: X
digit_caps/range/startConst*
_output_shapes
: *
dtype0*
value	B : X
digit_caps/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
digit_caps/rangeRangedigit_caps/range/start:output:0digit_caps/mod:z:0digit_caps/range/delta:output:0*
_output_shapes
:T
digit_caps/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :k
digit_caps/add_1AddV2digit_caps/mod:z:0digit_caps/add_1/y:output:0*
T0*
_output_shapes
: Z
digit_caps/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
digit_caps/range_1Rangedigit_caps/add_1:z:0digit_caps/Sub:z:0!digit_caps/range_1/delta:output:0*
_output_shapes
: d
digit_caps/concat/values_1Packdigit_caps/Sub:z:0*
N*
T0*
_output_shapes
:d
digit_caps/concat/values_3Packdigit_caps/mod:z:0*
N*
T0*
_output_shapes
:X
digit_caps/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
digit_caps/concatConcatV2digit_caps/range:output:0#digit_caps/concat/values_1:output:0digit_caps/range_1:output:0#digit_caps/concat/values_3:output:0digit_caps/concat/axis:output:0*
N*
T0*
_output_shapes
:?
digit_caps/transpose	Transposedigit_caps/Sum:output:0digit_caps/concat:output:0*
T0*/
_output_shapes
:?????????
q
digit_caps/SoftmaxSoftmaxdigit_caps/transpose:y:0*
T0*/
_output_shapes
:?????????
T
digit_caps/Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :q
digit_caps/Sub_1Subdigit_caps/Rank_1:output:0digit_caps/Sub_1/y:output:0*
T0*
_output_shapes
: Z
digit_caps/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : Z
digit_caps/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
digit_caps/range_2Range!digit_caps/range_2/start:output:0digit_caps/mod:z:0!digit_caps/range_2/delta:output:0*
_output_shapes
:T
digit_caps/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :k
digit_caps/add_2AddV2digit_caps/mod:z:0digit_caps/add_2/y:output:0*
T0*
_output_shapes
: Z
digit_caps/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
digit_caps/range_3Rangedigit_caps/add_2:z:0digit_caps/Sub_1:z:0!digit_caps/range_3/delta:output:0*
_output_shapes
: h
digit_caps/concat_1/values_1Packdigit_caps/Sub_1:z:0*
N*
T0*
_output_shapes
:f
digit_caps/concat_1/values_3Packdigit_caps/mod:z:0*
N*
T0*
_output_shapes
:Z
digit_caps/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
digit_caps/concat_1ConcatV2digit_caps/range_2:output:0%digit_caps/concat_1/values_1:output:0digit_caps/range_3:output:0%digit_caps/concat_1/values_3:output:0!digit_caps/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
*digit_caps/digit_cap_coupling_coefficients	Transposedigit_caps/Softmax:softmax:0digit_caps/concat_1:output:0*
T0*/
_output_shapes
:?????????
?
digit_caps/add_3/ReadVariableOpReadVariableOp(digit_caps_add_3_readvariableop_resource*"
_output_shapes
:
*
dtype0?
digit_caps/add_3AddV2.digit_caps/digit_cap_coupling_coefficients:y:0'digit_caps/add_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
?
digit_caps/MatMulBatchMatMulV2digit_caps/add_3:z:0)digit_caps/digit_cap_predictions:output:0*
T0*/
_output_shapes
:?????????
?
digit_caps/SqueezeSqueezedigit_caps/MatMul:output:0*
T0*+
_output_shapes
:?????????
*
squeeze_dims

??????????
$digit_caps/digit_cap_squash/norm/mulMuldigit_caps/Squeeze:output:0digit_caps/Squeeze:output:0*
T0*+
_output_shapes
:?????????
?
6digit_caps/digit_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
$digit_caps/digit_cap_squash/norm/SumSum(digit_caps/digit_cap_squash/norm/mul:z:0?digit_caps/digit_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????
*
	keep_dims(?
%digit_caps/digit_cap_squash/norm/SqrtSqrt-digit_caps/digit_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:?????????
?
digit_caps/digit_cap_squash/ExpExp)digit_caps/digit_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:?????????
j
%digit_caps/digit_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#digit_caps/digit_cap_squash/truedivRealDiv.digit_caps/digit_cap_squash/truediv/x:output:0#digit_caps/digit_cap_squash/Exp:y:0*
T0*+
_output_shapes
:?????????
f
!digit_caps/digit_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
digit_caps/digit_cap_squash/subSub*digit_caps/digit_cap_squash/sub/x:output:0'digit_caps/digit_cap_squash/truediv:z:0*
T0*+
_output_shapes
:?????????
f
!digit_caps/digit_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
digit_caps/digit_cap_squash/addAddV2)digit_caps/digit_cap_squash/norm/Sqrt:y:0*digit_caps/digit_cap_squash/add/y:output:0*
T0*+
_output_shapes
:?????????
?
%digit_caps/digit_cap_squash/truediv_1RealDivdigit_caps/Squeeze:output:0#digit_caps/digit_cap_squash/add:z:0*
T0*+
_output_shapes
:?????????
?
digit_caps/digit_cap_squash/mulMul#digit_caps/digit_cap_squash/sub:z:0)digit_caps/digit_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:?????????
?
digit_probs/norm/mulMul#digit_caps/digit_cap_squash/mul:z:0#digit_caps/digit_cap_squash/mul:z:0*
T0*+
_output_shapes
:?????????
y
&digit_probs/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
digit_probs/norm/SumSumdigit_probs/norm/mul:z:0/digit_probs/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????
*
	keep_dims(r
digit_probs/norm/SqrtSqrtdigit_probs/norm/Sum:output:0*
T0*+
_output_shapes
:?????????
?
digit_probs/norm/SqueezeSqueezedigit_probs/norm/Sqrt:y:0*
T0*'
_output_shapes
:?????????
*
squeeze_dims

?????????p
IdentityIdentity!digit_probs/norm/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp ^digit_caps/add_3/ReadVariableOp^digit_caps/map/while6^feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp5^feature_maps/feature_map_conv1/Conv2D/ReadVariableOp6^feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp5^feature_maps/feature_map_conv2/Conv2D/ReadVariableOp6^feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp5^feature_maps/feature_map_conv3/Conv2D/ReadVariableOp6^feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp5^feature_maps/feature_map_conv4/Conv2D/ReadVariableOp?^feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOpA^feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1.^feature_maps/feature_map_norm1/ReadVariableOp0^feature_maps/feature_map_norm1/ReadVariableOp_1?^feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOpA^feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1.^feature_maps/feature_map_norm2/ReadVariableOp0^feature_maps/feature_map_norm2/ReadVariableOp_1?^feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOpA^feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1.^feature_maps/feature_map_norm3/ReadVariableOp0^feature_maps/feature_map_norm3/ReadVariableOp_1?^feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOpA^feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1.^feature_maps/feature_map_norm4/ReadVariableOp0^feature_maps/feature_map_norm4/ReadVariableOp_16^primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp5^primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
digit_caps/add_3/ReadVariableOpdigit_caps/add_3/ReadVariableOp2,
digit_caps/map/whiledigit_caps/map/while2n
5feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp5feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp2l
4feature_maps/feature_map_conv1/Conv2D/ReadVariableOp4feature_maps/feature_map_conv1/Conv2D/ReadVariableOp2n
5feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp5feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp2l
4feature_maps/feature_map_conv2/Conv2D/ReadVariableOp4feature_maps/feature_map_conv2/Conv2D/ReadVariableOp2n
5feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp5feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp2l
4feature_maps/feature_map_conv3/Conv2D/ReadVariableOp4feature_maps/feature_map_conv3/Conv2D/ReadVariableOp2n
5feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp5feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp2l
4feature_maps/feature_map_conv4/Conv2D/ReadVariableOp4feature_maps/feature_map_conv4/Conv2D/ReadVariableOp2?
>feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp>feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp2?
@feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1@feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_12^
-feature_maps/feature_map_norm1/ReadVariableOp-feature_maps/feature_map_norm1/ReadVariableOp2b
/feature_maps/feature_map_norm1/ReadVariableOp_1/feature_maps/feature_map_norm1/ReadVariableOp_12?
>feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp>feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp2?
@feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1@feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_12^
-feature_maps/feature_map_norm2/ReadVariableOp-feature_maps/feature_map_norm2/ReadVariableOp2b
/feature_maps/feature_map_norm2/ReadVariableOp_1/feature_maps/feature_map_norm2/ReadVariableOp_12?
>feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp>feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp2?
@feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1@feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_12^
-feature_maps/feature_map_norm3/ReadVariableOp-feature_maps/feature_map_norm3/ReadVariableOp2b
/feature_maps/feature_map_norm3/ReadVariableOp_1/feature_maps/feature_map_norm3/ReadVariableOp_12?
>feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp>feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp2?
@feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1@feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_12^
-feature_maps/feature_map_norm4/ReadVariableOp-feature_maps/feature_map_norm4/ReadVariableOp2b
/feature_maps/feature_map_norm4/ReadVariableOp_1/feature_maps/feature_map_norm4/ReadVariableOp_12n
5primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp5primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp2l
4primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp4primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
M__inference_feature_map_norm1_layer_call_and_return_conditional_losses_724412

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
2__inference_feature_map_norm3_layer_call_fn_726976

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_feature_map_norm3_layer_call_and_return_conditional_losses_724540?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
2__inference_feature_map_norm3_layer_call_fn_726989

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_feature_map_norm3_layer_call_and_return_conditional_losses_724571?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?'
?
H__inference_primary_caps_layer_call_and_return_conditional_losses_724824
feature_mapsK
0primary_cap_dconv_conv2d_readvariableop_resource:		?@
1primary_cap_dconv_biasadd_readvariableop_resource:	?
identity??(primary_cap_dconv/BiasAdd/ReadVariableOp?'primary_cap_dconv/Conv2D/ReadVariableOp?
'primary_cap_dconv/Conv2D/ReadVariableOpReadVariableOp0primary_cap_dconv_conv2d_readvariableop_resource*'
_output_shapes
:		?*
dtype0?
primary_cap_dconv/Conv2DConv2Dfeature_maps/primary_cap_dconv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
(primary_cap_dconv/BiasAdd/ReadVariableOpReadVariableOp1primary_cap_dconv_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
primary_cap_dconv/BiasAddBiasAdd!primary_cap_dconv/Conv2D:output:00primary_cap_dconv/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????}
primary_cap_dconv/ReluRelu"primary_cap_dconv/BiasAdd:output:0*
T0*0
_output_shapes
:??????????m
primary_cap_reshape/ShapeShape$primary_cap_dconv/Relu:activations:0*
T0*
_output_shapes
:q
'primary_cap_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)primary_cap_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)primary_cap_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!primary_cap_reshape/strided_sliceStridedSlice"primary_cap_reshape/Shape:output:00primary_cap_reshape/strided_slice/stack:output:02primary_cap_reshape/strided_slice/stack_1:output:02primary_cap_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#primary_cap_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????e
#primary_cap_reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
!primary_cap_reshape/Reshape/shapePack*primary_cap_reshape/strided_slice:output:0,primary_cap_reshape/Reshape/shape/1:output:0,primary_cap_reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
primary_cap_reshape/ReshapeReshape$primary_cap_dconv/Relu:activations:0*primary_cap_reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:??????????
primary_cap_squash/norm/mulMul$primary_cap_reshape/Reshape:output:0$primary_cap_reshape/Reshape:output:0*
T0*+
_output_shapes
:??????????
-primary_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
primary_cap_squash/norm/SumSumprimary_cap_squash/norm/mul:z:06primary_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(?
primary_cap_squash/norm/SqrtSqrt$primary_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:?????????u
primary_cap_squash/ExpExp primary_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:?????????a
primary_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
primary_cap_squash/truedivRealDiv%primary_cap_squash/truediv/x:output:0primary_cap_squash/Exp:y:0*
T0*+
_output_shapes
:?????????]
primary_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
primary_cap_squash/subSub!primary_cap_squash/sub/x:output:0primary_cap_squash/truediv:z:0*
T0*+
_output_shapes
:?????????]
primary_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
primary_cap_squash/addAddV2 primary_cap_squash/norm/Sqrt:y:0!primary_cap_squash/add/y:output:0*
T0*+
_output_shapes
:??????????
primary_cap_squash/truediv_1RealDiv$primary_cap_reshape/Reshape:output:0primary_cap_squash/add:z:0*
T0*+
_output_shapes
:??????????
primary_cap_squash/mulMulprimary_cap_squash/sub:z:0 primary_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:?????????m
IdentityIdentityprimary_cap_squash/mul:z:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp)^primary_cap_dconv/BiasAdd/ReadVariableOp(^primary_cap_dconv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????		?: : 2T
(primary_cap_dconv/BiasAdd/ReadVariableOp(primary_cap_dconv/BiasAdd/ReadVariableOp2R
'primary_cap_dconv/Conv2D/ReadVariableOp'primary_cap_dconv/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:?????????		?
&
_user_specified_namefeature_maps
?
?
M__inference_feature_map_norm1_layer_call_and_return_conditional_losses_724443

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?!
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_726284

inputsW
=feature_maps_feature_map_conv1_conv2d_readvariableop_resource: L
>feature_maps_feature_map_conv1_biasadd_readvariableop_resource: D
6feature_maps_feature_map_norm1_readvariableop_resource: F
8feature_maps_feature_map_norm1_readvariableop_1_resource: U
Gfeature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_resource: W
Ifeature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource: W
=feature_maps_feature_map_conv2_conv2d_readvariableop_resource: @L
>feature_maps_feature_map_conv2_biasadd_readvariableop_resource:@D
6feature_maps_feature_map_norm2_readvariableop_resource:@F
8feature_maps_feature_map_norm2_readvariableop_1_resource:@U
Gfeature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_resource:@W
Ifeature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource:@W
=feature_maps_feature_map_conv3_conv2d_readvariableop_resource:@@L
>feature_maps_feature_map_conv3_biasadd_readvariableop_resource:@D
6feature_maps_feature_map_norm3_readvariableop_resource:@F
8feature_maps_feature_map_norm3_readvariableop_1_resource:@U
Gfeature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_resource:@W
Ifeature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource:@X
=feature_maps_feature_map_conv4_conv2d_readvariableop_resource:@?M
>feature_maps_feature_map_conv4_biasadd_readvariableop_resource:	?E
6feature_maps_feature_map_norm4_readvariableop_resource:	?G
8feature_maps_feature_map_norm4_readvariableop_1_resource:	?V
Gfeature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_resource:	?X
Ifeature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource:	?X
=primary_caps_primary_cap_dconv_conv2d_readvariableop_resource:		?M
>primary_caps_primary_cap_dconv_biasadd_readvariableop_resource:	?6
digit_caps_map_while_input_6:

digit_caps_mul_x>
(digit_caps_add_3_readvariableop_resource:

identity??digit_caps/add_3/ReadVariableOp?digit_caps/map/while?5feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp?4feature_maps/feature_map_conv1/Conv2D/ReadVariableOp?5feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp?4feature_maps/feature_map_conv2/Conv2D/ReadVariableOp?5feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp?4feature_maps/feature_map_conv3/Conv2D/ReadVariableOp?5feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp?4feature_maps/feature_map_conv4/Conv2D/ReadVariableOp?-feature_maps/feature_map_norm1/AssignNewValue?/feature_maps/feature_map_norm1/AssignNewValue_1?>feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp?@feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1?-feature_maps/feature_map_norm1/ReadVariableOp?/feature_maps/feature_map_norm1/ReadVariableOp_1?-feature_maps/feature_map_norm2/AssignNewValue?/feature_maps/feature_map_norm2/AssignNewValue_1?>feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp?@feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1?-feature_maps/feature_map_norm2/ReadVariableOp?/feature_maps/feature_map_norm2/ReadVariableOp_1?-feature_maps/feature_map_norm3/AssignNewValue?/feature_maps/feature_map_norm3/AssignNewValue_1?>feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp?@feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1?-feature_maps/feature_map_norm3/ReadVariableOp?/feature_maps/feature_map_norm3/ReadVariableOp_1?-feature_maps/feature_map_norm4/AssignNewValue?/feature_maps/feature_map_norm4/AssignNewValue_1?>feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp?@feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1?-feature_maps/feature_map_norm4/ReadVariableOp?/feature_maps/feature_map_norm4/ReadVariableOp_1?5primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp?4primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp?
4feature_maps/feature_map_conv1/Conv2D/ReadVariableOpReadVariableOp=feature_maps_feature_map_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
%feature_maps/feature_map_conv1/Conv2DConv2Dinputs<feature_maps/feature_map_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
5feature_maps/feature_map_conv1/BiasAdd/ReadVariableOpReadVariableOp>feature_maps_feature_map_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
&feature_maps/feature_map_conv1/BiasAddBiasAdd.feature_maps/feature_map_conv1/Conv2D:output:0=feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
#feature_maps/feature_map_conv1/ReluRelu/feature_maps/feature_map_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
-feature_maps/feature_map_norm1/ReadVariableOpReadVariableOp6feature_maps_feature_map_norm1_readvariableop_resource*
_output_shapes
: *
dtype0?
/feature_maps/feature_map_norm1/ReadVariableOp_1ReadVariableOp8feature_maps_feature_map_norm1_readvariableop_1_resource*
_output_shapes
: *
dtype0?
>feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOpReadVariableOpGfeature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
@feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIfeature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
/feature_maps/feature_map_norm1/FusedBatchNormV3FusedBatchNormV31feature_maps/feature_map_conv1/Relu:activations:05feature_maps/feature_map_norm1/ReadVariableOp:value:07feature_maps/feature_map_norm1/ReadVariableOp_1:value:0Ffeature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp:value:0Hfeature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
-feature_maps/feature_map_norm1/AssignNewValueAssignVariableOpGfeature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_resource<feature_maps/feature_map_norm1/FusedBatchNormV3:batch_mean:0?^feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
/feature_maps/feature_map_norm1/AssignNewValue_1AssignVariableOpIfeature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource@feature_maps/feature_map_norm1/FusedBatchNormV3:batch_variance:0A^feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
4feature_maps/feature_map_conv2/Conv2D/ReadVariableOpReadVariableOp=feature_maps_feature_map_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
%feature_maps/feature_map_conv2/Conv2DConv2D3feature_maps/feature_map_norm1/FusedBatchNormV3:y:0<feature_maps/feature_map_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
5feature_maps/feature_map_conv2/BiasAdd/ReadVariableOpReadVariableOp>feature_maps_feature_map_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
&feature_maps/feature_map_conv2/BiasAddBiasAdd.feature_maps/feature_map_conv2/Conv2D:output:0=feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
#feature_maps/feature_map_conv2/ReluRelu/feature_maps/feature_map_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
-feature_maps/feature_map_norm2/ReadVariableOpReadVariableOp6feature_maps_feature_map_norm2_readvariableop_resource*
_output_shapes
:@*
dtype0?
/feature_maps/feature_map_norm2/ReadVariableOp_1ReadVariableOp8feature_maps_feature_map_norm2_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
>feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOpReadVariableOpGfeature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
@feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIfeature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
/feature_maps/feature_map_norm2/FusedBatchNormV3FusedBatchNormV31feature_maps/feature_map_conv2/Relu:activations:05feature_maps/feature_map_norm2/ReadVariableOp:value:07feature_maps/feature_map_norm2/ReadVariableOp_1:value:0Ffeature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp:value:0Hfeature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
-feature_maps/feature_map_norm2/AssignNewValueAssignVariableOpGfeature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_resource<feature_maps/feature_map_norm2/FusedBatchNormV3:batch_mean:0?^feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
/feature_maps/feature_map_norm2/AssignNewValue_1AssignVariableOpIfeature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource@feature_maps/feature_map_norm2/FusedBatchNormV3:batch_variance:0A^feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
4feature_maps/feature_map_conv3/Conv2D/ReadVariableOpReadVariableOp=feature_maps_feature_map_conv3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
%feature_maps/feature_map_conv3/Conv2DConv2D3feature_maps/feature_map_norm2/FusedBatchNormV3:y:0<feature_maps/feature_map_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
5feature_maps/feature_map_conv3/BiasAdd/ReadVariableOpReadVariableOp>feature_maps_feature_map_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
&feature_maps/feature_map_conv3/BiasAddBiasAdd.feature_maps/feature_map_conv3/Conv2D:output:0=feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
#feature_maps/feature_map_conv3/ReluRelu/feature_maps/feature_map_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
-feature_maps/feature_map_norm3/ReadVariableOpReadVariableOp6feature_maps_feature_map_norm3_readvariableop_resource*
_output_shapes
:@*
dtype0?
/feature_maps/feature_map_norm3/ReadVariableOp_1ReadVariableOp8feature_maps_feature_map_norm3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
>feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOpReadVariableOpGfeature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
@feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIfeature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
/feature_maps/feature_map_norm3/FusedBatchNormV3FusedBatchNormV31feature_maps/feature_map_conv3/Relu:activations:05feature_maps/feature_map_norm3/ReadVariableOp:value:07feature_maps/feature_map_norm3/ReadVariableOp_1:value:0Ffeature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp:value:0Hfeature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
-feature_maps/feature_map_norm3/AssignNewValueAssignVariableOpGfeature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_resource<feature_maps/feature_map_norm3/FusedBatchNormV3:batch_mean:0?^feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
/feature_maps/feature_map_norm3/AssignNewValue_1AssignVariableOpIfeature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource@feature_maps/feature_map_norm3/FusedBatchNormV3:batch_variance:0A^feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
4feature_maps/feature_map_conv4/Conv2D/ReadVariableOpReadVariableOp=feature_maps_feature_map_conv4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
%feature_maps/feature_map_conv4/Conv2DConv2D3feature_maps/feature_map_norm3/FusedBatchNormV3:y:0<feature_maps/feature_map_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
?
5feature_maps/feature_map_conv4/BiasAdd/ReadVariableOpReadVariableOp>feature_maps_feature_map_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&feature_maps/feature_map_conv4/BiasAddBiasAdd.feature_maps/feature_map_conv4/Conv2D:output:0=feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		??
#feature_maps/feature_map_conv4/ReluRelu/feature_maps/feature_map_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		??
-feature_maps/feature_map_norm4/ReadVariableOpReadVariableOp6feature_maps_feature_map_norm4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
/feature_maps/feature_map_norm4/ReadVariableOp_1ReadVariableOp8feature_maps_feature_map_norm4_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
>feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOpReadVariableOpGfeature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
@feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIfeature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
/feature_maps/feature_map_norm4/FusedBatchNormV3FusedBatchNormV31feature_maps/feature_map_conv4/Relu:activations:05feature_maps/feature_map_norm4/ReadVariableOp:value:07feature_maps/feature_map_norm4/ReadVariableOp_1:value:0Ffeature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp:value:0Hfeature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????		?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
-feature_maps/feature_map_norm4/AssignNewValueAssignVariableOpGfeature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_resource<feature_maps/feature_map_norm4/FusedBatchNormV3:batch_mean:0?^feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
/feature_maps/feature_map_norm4/AssignNewValue_1AssignVariableOpIfeature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource@feature_maps/feature_map_norm4/FusedBatchNormV3:batch_variance:0A^feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
4primary_caps/primary_cap_dconv/Conv2D/ReadVariableOpReadVariableOp=primary_caps_primary_cap_dconv_conv2d_readvariableop_resource*'
_output_shapes
:		?*
dtype0?
%primary_caps/primary_cap_dconv/Conv2DConv2D3feature_maps/feature_map_norm4/FusedBatchNormV3:y:0<primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
5primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOpReadVariableOp>primary_caps_primary_cap_dconv_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&primary_caps/primary_cap_dconv/BiasAddBiasAdd.primary_caps/primary_cap_dconv/Conv2D:output:0=primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
#primary_caps/primary_cap_dconv/ReluRelu/primary_caps/primary_cap_dconv/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
&primary_caps/primary_cap_reshape/ShapeShape1primary_caps/primary_cap_dconv/Relu:activations:0*
T0*
_output_shapes
:~
4primary_caps/primary_cap_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6primary_caps/primary_cap_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6primary_caps/primary_cap_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.primary_caps/primary_cap_reshape/strided_sliceStridedSlice/primary_caps/primary_cap_reshape/Shape:output:0=primary_caps/primary_cap_reshape/strided_slice/stack:output:0?primary_caps/primary_cap_reshape/strided_slice/stack_1:output:0?primary_caps/primary_cap_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0primary_caps/primary_cap_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????r
0primary_caps/primary_cap_reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
.primary_caps/primary_cap_reshape/Reshape/shapePack7primary_caps/primary_cap_reshape/strided_slice:output:09primary_caps/primary_cap_reshape/Reshape/shape/1:output:09primary_caps/primary_cap_reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
(primary_caps/primary_cap_reshape/ReshapeReshape1primary_caps/primary_cap_dconv/Relu:activations:07primary_caps/primary_cap_reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:??????????
(primary_caps/primary_cap_squash/norm/mulMul1primary_caps/primary_cap_reshape/Reshape:output:01primary_caps/primary_cap_reshape/Reshape:output:0*
T0*+
_output_shapes
:??????????
:primary_caps/primary_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
(primary_caps/primary_cap_squash/norm/SumSum,primary_caps/primary_cap_squash/norm/mul:z:0Cprimary_caps/primary_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(?
)primary_caps/primary_cap_squash/norm/SqrtSqrt1primary_caps/primary_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:??????????
#primary_caps/primary_cap_squash/ExpExp-primary_caps/primary_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:?????????n
)primary_caps/primary_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'primary_caps/primary_cap_squash/truedivRealDiv2primary_caps/primary_cap_squash/truediv/x:output:0'primary_caps/primary_cap_squash/Exp:y:0*
T0*+
_output_shapes
:?????????j
%primary_caps/primary_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#primary_caps/primary_cap_squash/subSub.primary_caps/primary_cap_squash/sub/x:output:0+primary_caps/primary_cap_squash/truediv:z:0*
T0*+
_output_shapes
:?????????j
%primary_caps/primary_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
#primary_caps/primary_cap_squash/addAddV2-primary_caps/primary_cap_squash/norm/Sqrt:y:0.primary_caps/primary_cap_squash/add/y:output:0*
T0*+
_output_shapes
:??????????
)primary_caps/primary_cap_squash/truediv_1RealDiv1primary_caps/primary_cap_reshape/Reshape:output:0'primary_caps/primary_cap_squash/add:z:0*
T0*+
_output_shapes
:??????????
#primary_caps/primary_cap_squash/mulMul'primary_caps/primary_cap_squash/sub:z:0-primary_caps/primary_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:?????????[
digit_caps/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
digit_caps/ExpandDims
ExpandDims'primary_caps/primary_cap_squash/mul:z:0"digit_caps/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????r
digit_caps/Tile/multiplesConst*
_output_shapes
:*
dtype0*%
valueB"   
         ?
digit_caps/TileTiledigit_caps/ExpandDims:output:0"digit_caps/Tile/multiples:output:0*
T0*/
_output_shapes
:?????????
j
digit_caps/digit_cap_inputs/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
digit_caps/digit_cap_inputs
ExpandDimsdigit_caps/Tile:output:0(digit_caps/digit_cap_inputs/dim:output:0*
T0*3
_output_shapes!
:?????????
h
digit_caps/map/ShapeShape$digit_caps/digit_cap_inputs:output:0*
T0*
_output_shapes
:l
"digit_caps/map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$digit_caps/map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$digit_caps/map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
digit_caps/map/strided_sliceStridedSlicedigit_caps/map/Shape:output:0+digit_caps/map/strided_slice/stack:output:0-digit_caps/map/strided_slice/stack_1:output:0-digit_caps/map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
*digit_caps/map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
digit_caps/map/TensorArrayV2TensorListReserve3digit_caps/map/TensorArrayV2/element_shape:output:0%digit_caps/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Ddigit_caps/map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            ?
6digit_caps/map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$digit_caps/digit_cap_inputs:output:0Mdigit_caps/map/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???V
digit_caps/map/ConstConst*
_output_shapes
: *
dtype0*
value	B : w
,digit_caps/map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
digit_caps/map/TensorArrayV2_1TensorListReserve5digit_caps/map/TensorArrayV2_1/element_shape:output:0%digit_caps/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???c
!digit_caps/map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
digit_caps/map/whileWhile*digit_caps/map/while/loop_counter:output:0%digit_caps/map/strided_slice:output:0digit_caps/map/Const:output:0'digit_caps/map/TensorArrayV2_1:handle:0%digit_caps/map/strided_slice:output:0Fdigit_caps/map/TensorArrayUnstack/TensorListFromTensor:output_handle:0digit_caps_map_while_input_6*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : *#
_read_only_resource_inputs
*
_stateful_parallelism( *,
body$R"
 digit_caps_map_while_body_726174*,
cond$R"
 digit_caps_map_while_cond_726173*!
output_shapes
: : : : : : : ?
?digit_caps/map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            ?
1digit_caps/map/TensorArrayV2Stack/TensorListStackTensorListStackdigit_caps/map/while:output:3Hdigit_caps/map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:?????????
*
element_dtype0?
 digit_caps/digit_cap_predictionsSqueeze:digit_caps/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*/
_output_shapes
:?????????
*
squeeze_dims

??????????
digit_caps/digit_cap_attentionsBatchMatMulV2)digit_caps/digit_cap_predictions:output:0)digit_caps/digit_cap_predictions:output:0*
T0*/
_output_shapes
:?????????
*
adj_y(?
digit_caps/mulMuldigit_caps_mul_x(digit_caps/digit_cap_attentions:output:0*
T0*/
_output_shapes
:?????????
k
 digit_caps/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
digit_caps/SumSumdigit_caps/mul:z:0)digit_caps/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????
*
	keep_dims(Q
digit_caps/RankConst*
_output_shapes
: *
dtype0*
value	B :[
digit_caps/add/xConst*
_output_shapes
: *
dtype0*
valueB :
?????????m
digit_caps/addAddV2digit_caps/add/x:output:0digit_caps/Rank:output:0*
T0*
_output_shapes
: S
digit_caps/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :R
digit_caps/mod/yConst*
_output_shapes
: *
dtype0*
value	B :j
digit_caps/modFloorModdigit_caps/add:z:0digit_caps/mod/y:output:0*
T0*
_output_shapes
: R
digit_caps/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :m
digit_caps/SubSubdigit_caps/Rank_1:output:0digit_caps/Sub/y:output:0*
T0*
_output_shapes
: X
digit_caps/range/startConst*
_output_shapes
: *
dtype0*
value	B : X
digit_caps/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
digit_caps/rangeRangedigit_caps/range/start:output:0digit_caps/mod:z:0digit_caps/range/delta:output:0*
_output_shapes
:T
digit_caps/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :k
digit_caps/add_1AddV2digit_caps/mod:z:0digit_caps/add_1/y:output:0*
T0*
_output_shapes
: Z
digit_caps/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
digit_caps/range_1Rangedigit_caps/add_1:z:0digit_caps/Sub:z:0!digit_caps/range_1/delta:output:0*
_output_shapes
: d
digit_caps/concat/values_1Packdigit_caps/Sub:z:0*
N*
T0*
_output_shapes
:d
digit_caps/concat/values_3Packdigit_caps/mod:z:0*
N*
T0*
_output_shapes
:X
digit_caps/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
digit_caps/concatConcatV2digit_caps/range:output:0#digit_caps/concat/values_1:output:0digit_caps/range_1:output:0#digit_caps/concat/values_3:output:0digit_caps/concat/axis:output:0*
N*
T0*
_output_shapes
:?
digit_caps/transpose	Transposedigit_caps/Sum:output:0digit_caps/concat:output:0*
T0*/
_output_shapes
:?????????
q
digit_caps/SoftmaxSoftmaxdigit_caps/transpose:y:0*
T0*/
_output_shapes
:?????????
T
digit_caps/Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :q
digit_caps/Sub_1Subdigit_caps/Rank_1:output:0digit_caps/Sub_1/y:output:0*
T0*
_output_shapes
: Z
digit_caps/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : Z
digit_caps/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
digit_caps/range_2Range!digit_caps/range_2/start:output:0digit_caps/mod:z:0!digit_caps/range_2/delta:output:0*
_output_shapes
:T
digit_caps/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :k
digit_caps/add_2AddV2digit_caps/mod:z:0digit_caps/add_2/y:output:0*
T0*
_output_shapes
: Z
digit_caps/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
digit_caps/range_3Rangedigit_caps/add_2:z:0digit_caps/Sub_1:z:0!digit_caps/range_3/delta:output:0*
_output_shapes
: h
digit_caps/concat_1/values_1Packdigit_caps/Sub_1:z:0*
N*
T0*
_output_shapes
:f
digit_caps/concat_1/values_3Packdigit_caps/mod:z:0*
N*
T0*
_output_shapes
:Z
digit_caps/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
digit_caps/concat_1ConcatV2digit_caps/range_2:output:0%digit_caps/concat_1/values_1:output:0digit_caps/range_3:output:0%digit_caps/concat_1/values_3:output:0!digit_caps/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
*digit_caps/digit_cap_coupling_coefficients	Transposedigit_caps/Softmax:softmax:0digit_caps/concat_1:output:0*
T0*/
_output_shapes
:?????????
?
digit_caps/add_3/ReadVariableOpReadVariableOp(digit_caps_add_3_readvariableop_resource*"
_output_shapes
:
*
dtype0?
digit_caps/add_3AddV2.digit_caps/digit_cap_coupling_coefficients:y:0'digit_caps/add_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
?
digit_caps/MatMulBatchMatMulV2digit_caps/add_3:z:0)digit_caps/digit_cap_predictions:output:0*
T0*/
_output_shapes
:?????????
?
digit_caps/SqueezeSqueezedigit_caps/MatMul:output:0*
T0*+
_output_shapes
:?????????
*
squeeze_dims

??????????
$digit_caps/digit_cap_squash/norm/mulMuldigit_caps/Squeeze:output:0digit_caps/Squeeze:output:0*
T0*+
_output_shapes
:?????????
?
6digit_caps/digit_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
$digit_caps/digit_cap_squash/norm/SumSum(digit_caps/digit_cap_squash/norm/mul:z:0?digit_caps/digit_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????
*
	keep_dims(?
%digit_caps/digit_cap_squash/norm/SqrtSqrt-digit_caps/digit_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:?????????
?
digit_caps/digit_cap_squash/ExpExp)digit_caps/digit_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:?????????
j
%digit_caps/digit_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#digit_caps/digit_cap_squash/truedivRealDiv.digit_caps/digit_cap_squash/truediv/x:output:0#digit_caps/digit_cap_squash/Exp:y:0*
T0*+
_output_shapes
:?????????
f
!digit_caps/digit_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
digit_caps/digit_cap_squash/subSub*digit_caps/digit_cap_squash/sub/x:output:0'digit_caps/digit_cap_squash/truediv:z:0*
T0*+
_output_shapes
:?????????
f
!digit_caps/digit_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
digit_caps/digit_cap_squash/addAddV2)digit_caps/digit_cap_squash/norm/Sqrt:y:0*digit_caps/digit_cap_squash/add/y:output:0*
T0*+
_output_shapes
:?????????
?
%digit_caps/digit_cap_squash/truediv_1RealDivdigit_caps/Squeeze:output:0#digit_caps/digit_cap_squash/add:z:0*
T0*+
_output_shapes
:?????????
?
digit_caps/digit_cap_squash/mulMul#digit_caps/digit_cap_squash/sub:z:0)digit_caps/digit_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:?????????
?
digit_probs/norm/mulMul#digit_caps/digit_cap_squash/mul:z:0#digit_caps/digit_cap_squash/mul:z:0*
T0*+
_output_shapes
:?????????
y
&digit_probs/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
digit_probs/norm/SumSumdigit_probs/norm/mul:z:0/digit_probs/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????
*
	keep_dims(r
digit_probs/norm/SqrtSqrtdigit_probs/norm/Sum:output:0*
T0*+
_output_shapes
:?????????
?
digit_probs/norm/SqueezeSqueezedigit_probs/norm/Sqrt:y:0*
T0*'
_output_shapes
:?????????
*
squeeze_dims

?????????p
IdentityIdentity!digit_probs/norm/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp ^digit_caps/add_3/ReadVariableOp^digit_caps/map/while6^feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp5^feature_maps/feature_map_conv1/Conv2D/ReadVariableOp6^feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp5^feature_maps/feature_map_conv2/Conv2D/ReadVariableOp6^feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp5^feature_maps/feature_map_conv3/Conv2D/ReadVariableOp6^feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp5^feature_maps/feature_map_conv4/Conv2D/ReadVariableOp.^feature_maps/feature_map_norm1/AssignNewValue0^feature_maps/feature_map_norm1/AssignNewValue_1?^feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOpA^feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1.^feature_maps/feature_map_norm1/ReadVariableOp0^feature_maps/feature_map_norm1/ReadVariableOp_1.^feature_maps/feature_map_norm2/AssignNewValue0^feature_maps/feature_map_norm2/AssignNewValue_1?^feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOpA^feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1.^feature_maps/feature_map_norm2/ReadVariableOp0^feature_maps/feature_map_norm2/ReadVariableOp_1.^feature_maps/feature_map_norm3/AssignNewValue0^feature_maps/feature_map_norm3/AssignNewValue_1?^feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOpA^feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1.^feature_maps/feature_map_norm3/ReadVariableOp0^feature_maps/feature_map_norm3/ReadVariableOp_1.^feature_maps/feature_map_norm4/AssignNewValue0^feature_maps/feature_map_norm4/AssignNewValue_1?^feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOpA^feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1.^feature_maps/feature_map_norm4/ReadVariableOp0^feature_maps/feature_map_norm4/ReadVariableOp_16^primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp5^primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
digit_caps/add_3/ReadVariableOpdigit_caps/add_3/ReadVariableOp2,
digit_caps/map/whiledigit_caps/map/while2n
5feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp5feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp2l
4feature_maps/feature_map_conv1/Conv2D/ReadVariableOp4feature_maps/feature_map_conv1/Conv2D/ReadVariableOp2n
5feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp5feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp2l
4feature_maps/feature_map_conv2/Conv2D/ReadVariableOp4feature_maps/feature_map_conv2/Conv2D/ReadVariableOp2n
5feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp5feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp2l
4feature_maps/feature_map_conv3/Conv2D/ReadVariableOp4feature_maps/feature_map_conv3/Conv2D/ReadVariableOp2n
5feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp5feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp2l
4feature_maps/feature_map_conv4/Conv2D/ReadVariableOp4feature_maps/feature_map_conv4/Conv2D/ReadVariableOp2^
-feature_maps/feature_map_norm1/AssignNewValue-feature_maps/feature_map_norm1/AssignNewValue2b
/feature_maps/feature_map_norm1/AssignNewValue_1/feature_maps/feature_map_norm1/AssignNewValue_12?
>feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp>feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp2?
@feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1@feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_12^
-feature_maps/feature_map_norm1/ReadVariableOp-feature_maps/feature_map_norm1/ReadVariableOp2b
/feature_maps/feature_map_norm1/ReadVariableOp_1/feature_maps/feature_map_norm1/ReadVariableOp_12^
-feature_maps/feature_map_norm2/AssignNewValue-feature_maps/feature_map_norm2/AssignNewValue2b
/feature_maps/feature_map_norm2/AssignNewValue_1/feature_maps/feature_map_norm2/AssignNewValue_12?
>feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp>feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp2?
@feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1@feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_12^
-feature_maps/feature_map_norm2/ReadVariableOp-feature_maps/feature_map_norm2/ReadVariableOp2b
/feature_maps/feature_map_norm2/ReadVariableOp_1/feature_maps/feature_map_norm2/ReadVariableOp_12^
-feature_maps/feature_map_norm3/AssignNewValue-feature_maps/feature_map_norm3/AssignNewValue2b
/feature_maps/feature_map_norm3/AssignNewValue_1/feature_maps/feature_map_norm3/AssignNewValue_12?
>feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp>feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp2?
@feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1@feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_12^
-feature_maps/feature_map_norm3/ReadVariableOp-feature_maps/feature_map_norm3/ReadVariableOp2b
/feature_maps/feature_map_norm3/ReadVariableOp_1/feature_maps/feature_map_norm3/ReadVariableOp_12^
-feature_maps/feature_map_norm4/AssignNewValue-feature_maps/feature_map_norm4/AssignNewValue2b
/feature_maps/feature_map_norm4/AssignNewValue_1/feature_maps/feature_map_norm4/AssignNewValue_12?
>feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp>feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp2?
@feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1@feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_12^
-feature_maps/feature_map_norm4/ReadVariableOp-feature_maps/feature_map_norm4/ReadVariableOp2b
/feature_maps/feature_map_norm4/ReadVariableOp_1/feature_maps/feature_map_norm4/ReadVariableOp_12n
5primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp5primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp2l
4primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp4primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
c
G__inference_digit_probs_layer_call_and_return_conditional_losses_724974

inputs
identityU
norm/mulMulinputsinputs*
T0*+
_output_shapes
:?????????
m
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????
*
	keep_dims(Z
	norm/SqrtSqrtnorm/Sum:output:0*
T0*+
_output_shapes
:?????????
x
norm/SqueezeSqueezenorm/Sqrt:y:0*
T0*'
_output_shapes
:?????????
*
squeeze_dims

?????????]
IdentityIdentitynorm/Squeeze:output:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?"
?
 digit_caps_map_while_body_725929:
6digit_caps_map_while_digit_caps_map_while_loop_counter5
1digit_caps_map_while_digit_caps_map_strided_slice$
 digit_caps_map_while_placeholder&
"digit_caps_map_while_placeholder_19
5digit_caps_map_while_digit_caps_map_strided_slice_1_0u
qdigit_caps_map_while_tensorarrayv2read_tensorlistgetitem_digit_caps_map_tensorarrayunstack_tensorlistfromtensor_0O
5digit_caps_map_while_matmul_readvariableop_resource_0:
!
digit_caps_map_while_identity#
digit_caps_map_while_identity_1#
digit_caps_map_while_identity_2#
digit_caps_map_while_identity_37
3digit_caps_map_while_digit_caps_map_strided_slice_1s
odigit_caps_map_while_tensorarrayv2read_tensorlistgetitem_digit_caps_map_tensorarrayunstack_tensorlistfromtensorM
3digit_caps_map_while_matmul_readvariableop_resource:
??*digit_caps/map/while/MatMul/ReadVariableOp?
Fdigit_caps/map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            ?
8digit_caps/map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqdigit_caps_map_while_tensorarrayv2read_tensorlistgetitem_digit_caps_map_tensorarrayunstack_tensorlistfromtensor_0 digit_caps_map_while_placeholderOdigit_caps/map/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*&
_output_shapes
:
*
element_dtype0?
*digit_caps/map/while/MatMul/ReadVariableOpReadVariableOp5digit_caps_map_while_matmul_readvariableop_resource_0*&
_output_shapes
:
*
dtype0?
digit_caps/map/while/MatMulBatchMatMulV22digit_caps/map/while/MatMul/ReadVariableOp:value:0?digit_caps/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*&
_output_shapes
:
?
9digit_caps/map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"digit_caps_map_while_placeholder_1 digit_caps_map_while_placeholder$digit_caps/map/while/MatMul:output:0*
_output_shapes
: *
element_dtype0:???\
digit_caps/map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
digit_caps/map/while/addAddV2 digit_caps_map_while_placeholder#digit_caps/map/while/add/y:output:0*
T0*
_output_shapes
: ^
digit_caps/map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
digit_caps/map/while/add_1AddV26digit_caps_map_while_digit_caps_map_while_loop_counter%digit_caps/map/while/add_1/y:output:0*
T0*
_output_shapes
: ?
digit_caps/map/while/IdentityIdentitydigit_caps/map/while/add_1:z:0^digit_caps/map/while/NoOp*
T0*
_output_shapes
: ?
digit_caps/map/while/Identity_1Identity1digit_caps_map_while_digit_caps_map_strided_slice^digit_caps/map/while/NoOp*
T0*
_output_shapes
: ?
digit_caps/map/while/Identity_2Identitydigit_caps/map/while/add:z:0^digit_caps/map/while/NoOp*
T0*
_output_shapes
: ?
digit_caps/map/while/Identity_3IdentityIdigit_caps/map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^digit_caps/map/while/NoOp*
T0*
_output_shapes
: :????
digit_caps/map/while/NoOpNoOp+^digit_caps/map/while/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "l
3digit_caps_map_while_digit_caps_map_strided_slice_15digit_caps_map_while_digit_caps_map_strided_slice_1_0"G
digit_caps_map_while_identity&digit_caps/map/while/Identity:output:0"K
digit_caps_map_while_identity_1(digit_caps/map/while/Identity_1:output:0"K
digit_caps_map_while_identity_2(digit_caps/map/while/Identity_2:output:0"K
digit_caps_map_while_identity_3(digit_caps/map/while/Identity_3:output:0"l
3digit_caps_map_while_matmul_readvariableop_resource5digit_caps_map_while_matmul_readvariableop_resource_0"?
odigit_caps_map_while_tensorarrayv2read_tensorlistgetitem_digit_caps_map_tensorarrayunstack_tensorlistfromtensorqdigit_caps_map_while_tensorarrayv2read_tensorlistgetitem_digit_caps_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : : : 2X
*digit_caps/map/while/MatMul/ReadVariableOp*digit_caps/map/while/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
M__inference_feature_map_norm3_layer_call_and_return_conditional_losses_724540

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
2__inference_feature_map_norm4_layer_call_fn_727038

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_feature_map_norm4_layer_call_and_return_conditional_losses_724604?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_feature_map_norm3_layer_call_and_return_conditional_losses_727007

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?!
?

M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_724977

inputs-
feature_maps_724742: !
feature_maps_724744: !
feature_maps_724746: !
feature_maps_724748: !
feature_maps_724750: !
feature_maps_724752: -
feature_maps_724754: @!
feature_maps_724756:@!
feature_maps_724758:@!
feature_maps_724760:@!
feature_maps_724762:@!
feature_maps_724764:@-
feature_maps_724766:@@!
feature_maps_724768:@!
feature_maps_724770:@!
feature_maps_724772:@!
feature_maps_724774:@!
feature_maps_724776:@.
feature_maps_724778:@?"
feature_maps_724780:	?"
feature_maps_724782:	?"
feature_maps_724784:	?"
feature_maps_724786:	?"
feature_maps_724788:	?.
primary_caps_724825:		?"
primary_caps_724827:	?+
digit_caps_724958:

digit_caps_724960'
digit_caps_724962:

identity??"digit_caps/StatefulPartitionedCall?$feature_maps/StatefulPartitionedCall?$primary_caps/StatefulPartitionedCall?
$feature_maps/StatefulPartitionedCallStatefulPartitionedCallinputsfeature_maps_724742feature_maps_724744feature_maps_724746feature_maps_724748feature_maps_724750feature_maps_724752feature_maps_724754feature_maps_724756feature_maps_724758feature_maps_724760feature_maps_724762feature_maps_724764feature_maps_724766feature_maps_724768feature_maps_724770feature_maps_724772feature_maps_724774feature_maps_724776feature_maps_724778feature_maps_724780feature_maps_724782feature_maps_724784feature_maps_724786feature_maps_724788*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_feature_maps_layer_call_and_return_conditional_losses_724741?
$primary_caps/StatefulPartitionedCallStatefulPartitionedCall-feature_maps/StatefulPartitionedCall:output:0primary_caps_724825primary_caps_724827*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_primary_caps_layer_call_and_return_conditional_losses_724824?
"digit_caps/StatefulPartitionedCallStatefulPartitionedCall-primary_caps/StatefulPartitionedCall:output:0digit_caps_724958digit_caps_724960digit_caps_724962*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_digit_caps_layer_call_and_return_conditional_losses_724957?
digit_probs/PartitionedCallPartitionedCall+digit_caps/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_digit_probs_layer_call_and_return_conditional_losses_724974s
IdentityIdentity$digit_probs/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp#^digit_caps/StatefulPartitionedCall%^feature_maps/StatefulPartitionedCall%^primary_caps/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"digit_caps/StatefulPartitionedCall"digit_caps/StatefulPartitionedCall2L
$feature_maps/StatefulPartitionedCall$feature_maps/StatefulPartitionedCall2L
$primary_caps/StatefulPartitionedCall$primary_caps/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?m
?
H__inference_feature_maps_layer_call_and_return_conditional_losses_726543
input_imagesJ
0feature_map_conv1_conv2d_readvariableop_resource: ?
1feature_map_conv1_biasadd_readvariableop_resource: 7
)feature_map_norm1_readvariableop_resource: 9
+feature_map_norm1_readvariableop_1_resource: H
:feature_map_norm1_fusedbatchnormv3_readvariableop_resource: J
<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource: J
0feature_map_conv2_conv2d_readvariableop_resource: @?
1feature_map_conv2_biasadd_readvariableop_resource:@7
)feature_map_norm2_readvariableop_resource:@9
+feature_map_norm2_readvariableop_1_resource:@H
:feature_map_norm2_fusedbatchnormv3_readvariableop_resource:@J
<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource:@J
0feature_map_conv3_conv2d_readvariableop_resource:@@?
1feature_map_conv3_biasadd_readvariableop_resource:@7
)feature_map_norm3_readvariableop_resource:@9
+feature_map_norm3_readvariableop_1_resource:@H
:feature_map_norm3_fusedbatchnormv3_readvariableop_resource:@J
<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource:@K
0feature_map_conv4_conv2d_readvariableop_resource:@?@
1feature_map_conv4_biasadd_readvariableop_resource:	?8
)feature_map_norm4_readvariableop_resource:	?:
+feature_map_norm4_readvariableop_1_resource:	?I
:feature_map_norm4_fusedbatchnormv3_readvariableop_resource:	?K
<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource:	?
identity??(feature_map_conv1/BiasAdd/ReadVariableOp?'feature_map_conv1/Conv2D/ReadVariableOp?(feature_map_conv2/BiasAdd/ReadVariableOp?'feature_map_conv2/Conv2D/ReadVariableOp?(feature_map_conv3/BiasAdd/ReadVariableOp?'feature_map_conv3/Conv2D/ReadVariableOp?(feature_map_conv4/BiasAdd/ReadVariableOp?'feature_map_conv4/Conv2D/ReadVariableOp?1feature_map_norm1/FusedBatchNormV3/ReadVariableOp?3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1? feature_map_norm1/ReadVariableOp?"feature_map_norm1/ReadVariableOp_1?1feature_map_norm2/FusedBatchNormV3/ReadVariableOp?3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1? feature_map_norm2/ReadVariableOp?"feature_map_norm2/ReadVariableOp_1?1feature_map_norm3/FusedBatchNormV3/ReadVariableOp?3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1? feature_map_norm3/ReadVariableOp?"feature_map_norm3/ReadVariableOp_1?1feature_map_norm4/FusedBatchNormV3/ReadVariableOp?3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1? feature_map_norm4/ReadVariableOp?"feature_map_norm4/ReadVariableOp_1?
'feature_map_conv1/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
feature_map_conv1/Conv2DConv2Dinput_images/feature_map_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
(feature_map_conv1/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
feature_map_conv1/BiasAddBiasAdd!feature_map_conv1/Conv2D:output:00feature_map_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? |
feature_map_conv1/ReluRelu"feature_map_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
 feature_map_norm1/ReadVariableOpReadVariableOp)feature_map_norm1_readvariableop_resource*
_output_shapes
: *
dtype0?
"feature_map_norm1/ReadVariableOp_1ReadVariableOp+feature_map_norm1_readvariableop_1_resource*
_output_shapes
: *
dtype0?
1feature_map_norm1/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
"feature_map_norm1/FusedBatchNormV3FusedBatchNormV3$feature_map_conv1/Relu:activations:0(feature_map_norm1/ReadVariableOp:value:0*feature_map_norm1/ReadVariableOp_1:value:09feature_map_norm1/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
'feature_map_conv2/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
feature_map_conv2/Conv2DConv2D&feature_map_norm1/FusedBatchNormV3:y:0/feature_map_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
(feature_map_conv2/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
feature_map_conv2/BiasAddBiasAdd!feature_map_conv2/Conv2D:output:00feature_map_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@|
feature_map_conv2/ReluRelu"feature_map_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
 feature_map_norm2/ReadVariableOpReadVariableOp)feature_map_norm2_readvariableop_resource*
_output_shapes
:@*
dtype0?
"feature_map_norm2/ReadVariableOp_1ReadVariableOp+feature_map_norm2_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
1feature_map_norm2/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
"feature_map_norm2/FusedBatchNormV3FusedBatchNormV3$feature_map_conv2/Relu:activations:0(feature_map_norm2/ReadVariableOp:value:0*feature_map_norm2/ReadVariableOp_1:value:09feature_map_norm2/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
'feature_map_conv3/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
feature_map_conv3/Conv2DConv2D&feature_map_norm2/FusedBatchNormV3:y:0/feature_map_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
(feature_map_conv3/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
feature_map_conv3/BiasAddBiasAdd!feature_map_conv3/Conv2D:output:00feature_map_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@|
feature_map_conv3/ReluRelu"feature_map_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
 feature_map_norm3/ReadVariableOpReadVariableOp)feature_map_norm3_readvariableop_resource*
_output_shapes
:@*
dtype0?
"feature_map_norm3/ReadVariableOp_1ReadVariableOp+feature_map_norm3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
1feature_map_norm3/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
"feature_map_norm3/FusedBatchNormV3FusedBatchNormV3$feature_map_conv3/Relu:activations:0(feature_map_norm3/ReadVariableOp:value:0*feature_map_norm3/ReadVariableOp_1:value:09feature_map_norm3/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
'feature_map_conv4/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
feature_map_conv4/Conv2DConv2D&feature_map_norm3/FusedBatchNormV3:y:0/feature_map_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
?
(feature_map_conv4/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
feature_map_conv4/BiasAddBiasAdd!feature_map_conv4/Conv2D:output:00feature_map_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?}
feature_map_conv4/ReluRelu"feature_map_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		??
 feature_map_norm4/ReadVariableOpReadVariableOp)feature_map_norm4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"feature_map_norm4/ReadVariableOp_1ReadVariableOp+feature_map_norm4_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
1feature_map_norm4/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
"feature_map_norm4/FusedBatchNormV3FusedBatchNormV3$feature_map_conv4/Relu:activations:0(feature_map_norm4/ReadVariableOp:value:0*feature_map_norm4/ReadVariableOp_1:value:09feature_map_norm4/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????		?:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentity&feature_map_norm4/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????		??
NoOpNoOp)^feature_map_conv1/BiasAdd/ReadVariableOp(^feature_map_conv1/Conv2D/ReadVariableOp)^feature_map_conv2/BiasAdd/ReadVariableOp(^feature_map_conv2/Conv2D/ReadVariableOp)^feature_map_conv3/BiasAdd/ReadVariableOp(^feature_map_conv3/Conv2D/ReadVariableOp)^feature_map_conv4/BiasAdd/ReadVariableOp(^feature_map_conv4/Conv2D/ReadVariableOp2^feature_map_norm1/FusedBatchNormV3/ReadVariableOp4^feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm1/ReadVariableOp#^feature_map_norm1/ReadVariableOp_12^feature_map_norm2/FusedBatchNormV3/ReadVariableOp4^feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm2/ReadVariableOp#^feature_map_norm2/ReadVariableOp_12^feature_map_norm3/FusedBatchNormV3/ReadVariableOp4^feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm3/ReadVariableOp#^feature_map_norm3/ReadVariableOp_12^feature_map_norm4/FusedBatchNormV3/ReadVariableOp4^feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm4/ReadVariableOp#^feature_map_norm4/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????: : : : : : : : : : : : : : : : : : : : : : : : 2T
(feature_map_conv1/BiasAdd/ReadVariableOp(feature_map_conv1/BiasAdd/ReadVariableOp2R
'feature_map_conv1/Conv2D/ReadVariableOp'feature_map_conv1/Conv2D/ReadVariableOp2T
(feature_map_conv2/BiasAdd/ReadVariableOp(feature_map_conv2/BiasAdd/ReadVariableOp2R
'feature_map_conv2/Conv2D/ReadVariableOp'feature_map_conv2/Conv2D/ReadVariableOp2T
(feature_map_conv3/BiasAdd/ReadVariableOp(feature_map_conv3/BiasAdd/ReadVariableOp2R
'feature_map_conv3/Conv2D/ReadVariableOp'feature_map_conv3/Conv2D/ReadVariableOp2T
(feature_map_conv4/BiasAdd/ReadVariableOp(feature_map_conv4/BiasAdd/ReadVariableOp2R
'feature_map_conv4/Conv2D/ReadVariableOp'feature_map_conv4/Conv2D/ReadVariableOp2f
1feature_map_norm1/FusedBatchNormV3/ReadVariableOp1feature_map_norm1/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_13feature_map_norm1/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm1/ReadVariableOp feature_map_norm1/ReadVariableOp2H
"feature_map_norm1/ReadVariableOp_1"feature_map_norm1/ReadVariableOp_12f
1feature_map_norm2/FusedBatchNormV3/ReadVariableOp1feature_map_norm2/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_13feature_map_norm2/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm2/ReadVariableOp feature_map_norm2/ReadVariableOp2H
"feature_map_norm2/ReadVariableOp_1"feature_map_norm2/ReadVariableOp_12f
1feature_map_norm3/FusedBatchNormV3/ReadVariableOp1feature_map_norm3/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_13feature_map_norm3/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm3/ReadVariableOp feature_map_norm3/ReadVariableOp2H
"feature_map_norm3/ReadVariableOp_1"feature_map_norm3/ReadVariableOp_12f
1feature_map_norm4/FusedBatchNormV3/ReadVariableOp1feature_map_norm4/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_13feature_map_norm4/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm4/ReadVariableOp feature_map_norm4/ReadVariableOp2H
"feature_map_norm4/ReadVariableOp_1"feature_map_norm4/ReadVariableOp_1:] Y
/
_output_shapes
:?????????
&
_user_specified_nameinput_images
?
?
M__inference_feature_map_norm2_layer_call_and_return_conditional_losses_726963

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
M
input_images=
serving_default_input_images:0??????????
digit_probs0
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
6
_init_input_shape"
_tf_keras_input_layer
?
	conv1
	norm1
	conv2
	norm2
	conv3
	norm3
	conv4
	norm4
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	dconv
reshape

 squash
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
?
'digit_caps_transform_tensor
'W
(digit_caps_log_priors
(B

)squash
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
?
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
?
6iter

7beta_1

8beta_2
	9decay
:learning_rate'm?(m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?Sm?Tm?'v?(v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?Sv?Tv?"
	optimizer
?
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15
K16
L17
M18
N19
O20
P21
Q22
R23
S24
T25
'26
(27"
trackable_list_wrapper
?
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15
S16
T17
'18
(19"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
2__inference_Efficient-CapsNet_layer_call_fn_725038
2__inference_Efficient-CapsNet_layer_call_fn_725731
2__inference_Efficient-CapsNet_layer_call_fn_725794
2__inference_Efficient-CapsNet_layer_call_fn_725530?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_726039
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_726284
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_725596
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_725662?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_724390input_images"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
Zserving_default"
signature_map
 "
trackable_list_wrapper
?

;kernel
<bias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
?
aaxis
	=gamma
>beta
Kmoving_mean
Lmoving_variance
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses"
_tf_keras_layer
?

?kernel
@bias
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_layer
?
naxis
	Agamma
Bbeta
Mmoving_mean
Nmoving_variance
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Ckernel
Dbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses"
_tf_keras_layer
?
{axis
	Egamma
Fbeta
Omoving_mean
Pmoving_variance
|	variables
}trainable_variables
~regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Gkernel
Hbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	Igamma
Jbeta
Qmoving_mean
Rmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15
K16
L17
M18
N19
O20
P21
Q22
R23"
trackable_list_wrapper
?
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_feature_maps_layer_call_fn_726402
-__inference_feature_maps_layer_call_fn_726455?
???
FullArgSpec/
args'?$
jself
jinput_images

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_feature_maps_layer_call_and_return_conditional_losses_726543
H__inference_feature_maps_layer_call_and_return_conditional_losses_726631?
???
FullArgSpec/
args'?$
jself
jinput_images

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?

Skernel
Tbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_primary_caps_layer_call_fn_726640?
???
FullArgSpec#
args?
jself
jfeature_maps
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_primary_caps_layer_call_and_return_conditional_losses_726673?
???
FullArgSpec#
args?
jself
jfeature_maps
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
@:>
2&digit_caps/digit_caps_transform_tensor
6:4
2 digit_caps/digit_caps_log_priors
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_digit_caps_layer_call_fn_726684?
???
FullArgSpec#
args?
jself
jprimary_caps
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_digit_caps_layer_call_and_return_conditional_losses_726811?
???
FullArgSpec#
args?
jself
jprimary_caps
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_digit_probs_layer_call_fn_726816
,__inference_digit_probs_layer_call_fn_726821?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_digit_probs_layer_call_and_return_conditional_losses_726830
G__inference_digit_probs_layer_call_and_return_conditional_losses_726839?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?:= 2%feature_maps/feature_map_conv1/kernel
1:/ 2#feature_maps/feature_map_conv1/bias
2:0 2$feature_maps/feature_map_norm1/gamma
1:/ 2#feature_maps/feature_map_norm1/beta
?:= @2%feature_maps/feature_map_conv2/kernel
1:/@2#feature_maps/feature_map_conv2/bias
2:0@2$feature_maps/feature_map_norm2/gamma
1:/@2#feature_maps/feature_map_norm2/beta
?:=@@2%feature_maps/feature_map_conv3/kernel
1:/@2#feature_maps/feature_map_conv3/bias
2:0@2$feature_maps/feature_map_norm3/gamma
1:/@2#feature_maps/feature_map_norm3/beta
@:>@?2%feature_maps/feature_map_conv4/kernel
2:0?2#feature_maps/feature_map_conv4/bias
3:1?2$feature_maps/feature_map_norm4/gamma
2:0?2#feature_maps/feature_map_norm4/beta
::8  (2*feature_maps/feature_map_norm1/moving_mean
>:<  (2.feature_maps/feature_map_norm1/moving_variance
::8@ (2*feature_maps/feature_map_norm2/moving_mean
>:<@ (2.feature_maps/feature_map_norm2/moving_variance
::8@ (2*feature_maps/feature_map_norm3/moving_mean
>:<@ (2.feature_maps/feature_map_norm3/moving_variance
;:9? (2*feature_maps/feature_map_norm4/moving_mean
?:=? (2.feature_maps/feature_map_norm4/moving_variance
@:>		?2%primary_caps/primary_cap_dconv/kernel
2:0?2#primary_caps/primary_cap_dconv/bias
X
K0
L1
M2
N3
O4
P5
Q6
R7"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
$__inference_signature_wrapper_726349input_images"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
<
=0
>1
K2
L3"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
?2?
2__inference_feature_map_norm1_layer_call_fn_726852
2__inference_feature_map_norm1_layer_call_fn_726865?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_feature_map_norm1_layer_call_and_return_conditional_losses_726883
M__inference_feature_map_norm1_layer_call_and_return_conditional_losses_726901?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
<
A0
B1
M2
N3"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
?2?
2__inference_feature_map_norm2_layer_call_fn_726914
2__inference_feature_map_norm2_layer_call_fn_726927?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_feature_map_norm2_layer_call_and_return_conditional_losses_726945
M__inference_feature_map_norm2_layer_call_and_return_conditional_losses_726963?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
<
E0
F1
O2
P3"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
|	variables
}trainable_variables
~regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
2__inference_feature_map_norm3_layer_call_fn_726976
2__inference_feature_map_norm3_layer_call_fn_726989?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_feature_map_norm3_layer_call_and_return_conditional_losses_727007
M__inference_feature_map_norm3_layer_call_and_return_conditional_losses_727025?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
<
I0
J1
Q2
R3"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
2__inference_feature_map_norm4_layer_call_fn_727038
2__inference_feature_map_norm4_layer_call_fn_727051?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_feature_map_norm4_layer_call_and_return_conditional_losses_727069
M__inference_feature_map_norm4_layer_call_and_return_conditional_losses_727087?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
X
K0
L1
M2
N3
O4
P5
Q6
R7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec#
args?
jself
jinput_vector
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec#
args?
jself
jinput_vector
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
5
0
1
 2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec#
args?
jself
jinput_vector
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec#
args?
jself
jinput_vector
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
'
)0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
E:C
2-Adam/digit_caps/digit_caps_transform_tensor/m
;:9
2'Adam/digit_caps/digit_caps_log_priors/m
D:B 2,Adam/feature_maps/feature_map_conv1/kernel/m
6:4 2*Adam/feature_maps/feature_map_conv1/bias/m
7:5 2+Adam/feature_maps/feature_map_norm1/gamma/m
6:4 2*Adam/feature_maps/feature_map_norm1/beta/m
D:B @2,Adam/feature_maps/feature_map_conv2/kernel/m
6:4@2*Adam/feature_maps/feature_map_conv2/bias/m
7:5@2+Adam/feature_maps/feature_map_norm2/gamma/m
6:4@2*Adam/feature_maps/feature_map_norm2/beta/m
D:B@@2,Adam/feature_maps/feature_map_conv3/kernel/m
6:4@2*Adam/feature_maps/feature_map_conv3/bias/m
7:5@2+Adam/feature_maps/feature_map_norm3/gamma/m
6:4@2*Adam/feature_maps/feature_map_norm3/beta/m
E:C@?2,Adam/feature_maps/feature_map_conv4/kernel/m
7:5?2*Adam/feature_maps/feature_map_conv4/bias/m
8:6?2+Adam/feature_maps/feature_map_norm4/gamma/m
7:5?2*Adam/feature_maps/feature_map_norm4/beta/m
E:C		?2,Adam/primary_caps/primary_cap_dconv/kernel/m
7:5?2*Adam/primary_caps/primary_cap_dconv/bias/m
E:C
2-Adam/digit_caps/digit_caps_transform_tensor/v
;:9
2'Adam/digit_caps/digit_caps_log_priors/v
D:B 2,Adam/feature_maps/feature_map_conv1/kernel/v
6:4 2*Adam/feature_maps/feature_map_conv1/bias/v
7:5 2+Adam/feature_maps/feature_map_norm1/gamma/v
6:4 2*Adam/feature_maps/feature_map_norm1/beta/v
D:B @2,Adam/feature_maps/feature_map_conv2/kernel/v
6:4@2*Adam/feature_maps/feature_map_conv2/bias/v
7:5@2+Adam/feature_maps/feature_map_norm2/gamma/v
6:4@2*Adam/feature_maps/feature_map_norm2/beta/v
D:B@@2,Adam/feature_maps/feature_map_conv3/kernel/v
6:4@2*Adam/feature_maps/feature_map_conv3/bias/v
7:5@2+Adam/feature_maps/feature_map_norm3/gamma/v
6:4@2*Adam/feature_maps/feature_map_norm3/beta/v
E:C@?2,Adam/feature_maps/feature_map_conv4/kernel/v
7:5?2*Adam/feature_maps/feature_map_conv4/bias/v
8:6?2+Adam/feature_maps/feature_map_norm4/gamma/v
7:5?2*Adam/feature_maps/feature_map_norm4/beta/v
E:C		?2,Adam/primary_caps/primary_cap_dconv/kernel/v
7:5?2*Adam/primary_caps/primary_cap_dconv/bias/v
	J
Const?
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_725596?;<=>KL?@ABMNCDEFOPGHIJQRST'?(E?B
;?8
.?+
input_images?????????
p 

 
? "%?"
?
0?????????

? ?
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_725662?;<=>KL?@ABMNCDEFOPGHIJQRST'?(E?B
;?8
.?+
input_images?????????
p

 
? "%?"
?
0?????????

? ?
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_726039?;<=>KL?@ABMNCDEFOPGHIJQRST'?(??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_726284?;<=>KL?@ABMNCDEFOPGHIJQRST'?(??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????

? ?
2__inference_Efficient-CapsNet_layer_call_fn_725038?;<=>KL?@ABMNCDEFOPGHIJQRST'?(E?B
;?8
.?+
input_images?????????
p 

 
? "??????????
?
2__inference_Efficient-CapsNet_layer_call_fn_725530?;<=>KL?@ABMNCDEFOPGHIJQRST'?(E?B
;?8
.?+
input_images?????????
p

 
? "??????????
?
2__inference_Efficient-CapsNet_layer_call_fn_725731{;<=>KL?@ABMNCDEFOPGHIJQRST'?(??<
5?2
(?%
inputs?????????
p 

 
? "??????????
?
2__inference_Efficient-CapsNet_layer_call_fn_725794{;<=>KL?@ABMNCDEFOPGHIJQRST'?(??<
5?2
(?%
inputs?????????
p

 
? "??????????
?
!__inference__wrapped_model_724390?;<=>KL?@ABMNCDEFOPGHIJQRST'?(=?:
3?0
.?+
input_images?????????
? "9?6
4
digit_probs%?"
digit_probs?????????
?
F__inference_digit_caps_layer_call_and_return_conditional_losses_726811l'?(9?6
/?,
*?'
primary_caps?????????
? ")?&
?
0?????????

? ?
+__inference_digit_caps_layer_call_fn_726684_'?(9?6
/?,
*?'
primary_caps?????????
? "??????????
?
G__inference_digit_probs_layer_call_and_return_conditional_losses_726830d;?8
1?.
$?!
inputs?????????


 
p 
? "%?"
?
0?????????

? ?
G__inference_digit_probs_layer_call_and_return_conditional_losses_726839d;?8
1?.
$?!
inputs?????????


 
p
? "%?"
?
0?????????

? ?
,__inference_digit_probs_layer_call_fn_726816W;?8
1?.
$?!
inputs?????????


 
p 
? "??????????
?
,__inference_digit_probs_layer_call_fn_726821W;?8
1?.
$?!
inputs?????????


 
p
? "??????????
?
M__inference_feature_map_norm1_layer_call_and_return_conditional_losses_726883?=>KLM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
M__inference_feature_map_norm1_layer_call_and_return_conditional_losses_726901?=>KLM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
2__inference_feature_map_norm1_layer_call_fn_726852?=>KLM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
2__inference_feature_map_norm1_layer_call_fn_726865?=>KLM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
M__inference_feature_map_norm2_layer_call_and_return_conditional_losses_726945?ABMNM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
M__inference_feature_map_norm2_layer_call_and_return_conditional_losses_726963?ABMNM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
2__inference_feature_map_norm2_layer_call_fn_726914?ABMNM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
2__inference_feature_map_norm2_layer_call_fn_726927?ABMNM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
M__inference_feature_map_norm3_layer_call_and_return_conditional_losses_727007?EFOPM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
M__inference_feature_map_norm3_layer_call_and_return_conditional_losses_727025?EFOPM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
2__inference_feature_map_norm3_layer_call_fn_726976?EFOPM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
2__inference_feature_map_norm3_layer_call_fn_726989?EFOPM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
M__inference_feature_map_norm4_layer_call_and_return_conditional_losses_727069?IJQRN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
M__inference_feature_map_norm4_layer_call_and_return_conditional_losses_727087?IJQRN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
2__inference_feature_map_norm4_layer_call_fn_727038?IJQRN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
2__inference_feature_map_norm4_layer_call_fn_727051?IJQRN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
H__inference_feature_maps_layer_call_and_return_conditional_losses_726543?;<=>KL?@ABMNCDEFOPGHIJQRA?>
7?4
.?+
input_images?????????
p 
? ".?+
$?!
0?????????		?
? ?
H__inference_feature_maps_layer_call_and_return_conditional_losses_726631?;<=>KL?@ABMNCDEFOPGHIJQRA?>
7?4
.?+
input_images?????????
p
? ".?+
$?!
0?????????		?
? ?
-__inference_feature_maps_layer_call_fn_726402?;<=>KL?@ABMNCDEFOPGHIJQRA?>
7?4
.?+
input_images?????????
p 
? "!??????????		??
-__inference_feature_maps_layer_call_fn_726455?;<=>KL?@ABMNCDEFOPGHIJQRA?>
7?4
.?+
input_images?????????
p
? "!??????????		??
H__inference_primary_caps_layer_call_and_return_conditional_losses_726673oST>?;
4?1
/?,
feature_maps?????????		?
? ")?&
?
0?????????
? ?
-__inference_primary_caps_layer_call_fn_726640bST>?;
4?1
/?,
feature_maps?????????		?
? "???????????
$__inference_signature_wrapper_726349?;<=>KL?@ABMNCDEFOPGHIJQRST'?(M?J
? 
C?@
>
input_images.?+
input_images?????????"9?6
4
digit_probs%?"
digit_probs?????????
