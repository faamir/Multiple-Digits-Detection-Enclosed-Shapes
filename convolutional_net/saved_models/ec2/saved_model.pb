Ä
¦$÷#
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

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
ú
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
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
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

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

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
"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Î
°
&digit_caps/digit_caps_transform_tensorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*7
shared_name(&digit_caps/digit_caps_transform_tensor
©
:digit_caps/digit_caps_transform_tensor/Read/ReadVariableOpReadVariableOp&digit_caps/digit_caps_transform_tensor*&
_output_shapes
:
*
dtype0
 
 digit_caps/digit_caps_log_priorsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" digit_caps/digit_caps_log_priors

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
®
%feature_maps/feature_map_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%feature_maps/feature_map_conv1/kernel
§
9feature_maps/feature_map_conv1/kernel/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv1/kernel*&
_output_shapes
: *
dtype0

#feature_maps/feature_map_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#feature_maps/feature_map_conv1/bias

7feature_maps/feature_map_conv1/bias/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_conv1/bias*
_output_shapes
: *
dtype0
 
$feature_maps/feature_map_norm1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$feature_maps/feature_map_norm1/gamma

8feature_maps/feature_map_norm1/gamma/Read/ReadVariableOpReadVariableOp$feature_maps/feature_map_norm1/gamma*
_output_shapes
: *
dtype0

#feature_maps/feature_map_norm1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#feature_maps/feature_map_norm1/beta

7feature_maps/feature_map_norm1/beta/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_norm1/beta*
_output_shapes
: *
dtype0
®
%feature_maps/feature_map_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*6
shared_name'%feature_maps/feature_map_conv2/kernel
§
9feature_maps/feature_map_conv2/kernel/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv2/kernel*&
_output_shapes
: @*
dtype0

#feature_maps/feature_map_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#feature_maps/feature_map_conv2/bias

7feature_maps/feature_map_conv2/bias/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_conv2/bias*
_output_shapes
:@*
dtype0
 
$feature_maps/feature_map_norm2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$feature_maps/feature_map_norm2/gamma

8feature_maps/feature_map_norm2/gamma/Read/ReadVariableOpReadVariableOp$feature_maps/feature_map_norm2/gamma*
_output_shapes
:@*
dtype0

#feature_maps/feature_map_norm2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#feature_maps/feature_map_norm2/beta

7feature_maps/feature_map_norm2/beta/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_norm2/beta*
_output_shapes
:@*
dtype0
®
%feature_maps/feature_map_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*6
shared_name'%feature_maps/feature_map_conv3/kernel
§
9feature_maps/feature_map_conv3/kernel/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv3/kernel*&
_output_shapes
:@@*
dtype0

#feature_maps/feature_map_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#feature_maps/feature_map_conv3/bias

7feature_maps/feature_map_conv3/bias/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_conv3/bias*
_output_shapes
:@*
dtype0
 
$feature_maps/feature_map_norm3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$feature_maps/feature_map_norm3/gamma

8feature_maps/feature_map_norm3/gamma/Read/ReadVariableOpReadVariableOp$feature_maps/feature_map_norm3/gamma*
_output_shapes
:@*
dtype0

#feature_maps/feature_map_norm3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#feature_maps/feature_map_norm3/beta

7feature_maps/feature_map_norm3/beta/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_norm3/beta*
_output_shapes
:@*
dtype0
¯
%feature_maps/feature_map_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%feature_maps/feature_map_conv4/kernel
¨
9feature_maps/feature_map_conv4/kernel/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv4/kernel*'
_output_shapes
:@*
dtype0

#feature_maps/feature_map_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#feature_maps/feature_map_conv4/bias

7feature_maps/feature_map_conv4/bias/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_conv4/bias*
_output_shapes	
:*
dtype0
¡
$feature_maps/feature_map_norm4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$feature_maps/feature_map_norm4/gamma

8feature_maps/feature_map_norm4/gamma/Read/ReadVariableOpReadVariableOp$feature_maps/feature_map_norm4/gamma*
_output_shapes	
:*
dtype0

#feature_maps/feature_map_norm4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#feature_maps/feature_map_norm4/beta

7feature_maps/feature_map_norm4/beta/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_norm4/beta*
_output_shapes	
:*
dtype0
¬
*feature_maps/feature_map_norm1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*feature_maps/feature_map_norm1/moving_mean
¥
>feature_maps/feature_map_norm1/moving_mean/Read/ReadVariableOpReadVariableOp*feature_maps/feature_map_norm1/moving_mean*
_output_shapes
: *
dtype0
´
.feature_maps/feature_map_norm1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.feature_maps/feature_map_norm1/moving_variance
­
Bfeature_maps/feature_map_norm1/moving_variance/Read/ReadVariableOpReadVariableOp.feature_maps/feature_map_norm1/moving_variance*
_output_shapes
: *
dtype0
¬
*feature_maps/feature_map_norm2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*feature_maps/feature_map_norm2/moving_mean
¥
>feature_maps/feature_map_norm2/moving_mean/Read/ReadVariableOpReadVariableOp*feature_maps/feature_map_norm2/moving_mean*
_output_shapes
:@*
dtype0
´
.feature_maps/feature_map_norm2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.feature_maps/feature_map_norm2/moving_variance
­
Bfeature_maps/feature_map_norm2/moving_variance/Read/ReadVariableOpReadVariableOp.feature_maps/feature_map_norm2/moving_variance*
_output_shapes
:@*
dtype0
¬
*feature_maps/feature_map_norm3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*feature_maps/feature_map_norm3/moving_mean
¥
>feature_maps/feature_map_norm3/moving_mean/Read/ReadVariableOpReadVariableOp*feature_maps/feature_map_norm3/moving_mean*
_output_shapes
:@*
dtype0
´
.feature_maps/feature_map_norm3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.feature_maps/feature_map_norm3/moving_variance
­
Bfeature_maps/feature_map_norm3/moving_variance/Read/ReadVariableOpReadVariableOp.feature_maps/feature_map_norm3/moving_variance*
_output_shapes
:@*
dtype0
­
*feature_maps/feature_map_norm4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*feature_maps/feature_map_norm4/moving_mean
¦
>feature_maps/feature_map_norm4/moving_mean/Read/ReadVariableOpReadVariableOp*feature_maps/feature_map_norm4/moving_mean*
_output_shapes	
:*
dtype0
µ
.feature_maps/feature_map_norm4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.feature_maps/feature_map_norm4/moving_variance
®
Bfeature_maps/feature_map_norm4/moving_variance/Read/ReadVariableOpReadVariableOp.feature_maps/feature_map_norm4/moving_variance*
_output_shapes	
:*
dtype0
¯
%primary_caps/primary_cap_dconv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*6
shared_name'%primary_caps/primary_cap_dconv/kernel
¨
9primary_caps/primary_cap_dconv/kernel/Read/ReadVariableOpReadVariableOp%primary_caps/primary_cap_dconv/kernel*'
_output_shapes
:		*
dtype0

#primary_caps/primary_cap_dconv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#primary_caps/primary_cap_dconv/bias

7primary_caps/primary_cap_dconv/bias/Read/ReadVariableOpReadVariableOp#primary_caps/primary_cap_dconv/bias*
_output_shapes	
:*
dtype0
´
(digit_caps/digit_caps_transform_tensor/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(digit_caps/digit_caps_transform_tensor/m
­
<digit_caps/digit_caps_transform_tensor/m/Read/ReadVariableOpReadVariableOp(digit_caps/digit_caps_transform_tensor/m*&
_output_shapes
:
*
dtype0
¤
"digit_caps/digit_caps_log_priors/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"digit_caps/digit_caps_log_priors/m

6digit_caps/digit_caps_log_priors/m/Read/ReadVariableOpReadVariableOp"digit_caps/digit_caps_log_priors/m*"
_output_shapes
:
*
dtype0
²
'feature_maps/feature_map_conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'feature_maps/feature_map_conv1/kernel/m
«
;feature_maps/feature_map_conv1/kernel/m/Read/ReadVariableOpReadVariableOp'feature_maps/feature_map_conv1/kernel/m*&
_output_shapes
: *
dtype0
¢
%feature_maps/feature_map_conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%feature_maps/feature_map_conv1/bias/m

9feature_maps/feature_map_conv1/bias/m/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv1/bias/m*
_output_shapes
: *
dtype0
¤
&feature_maps/feature_map_norm1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&feature_maps/feature_map_norm1/gamma/m

:feature_maps/feature_map_norm1/gamma/m/Read/ReadVariableOpReadVariableOp&feature_maps/feature_map_norm1/gamma/m*
_output_shapes
: *
dtype0
¢
%feature_maps/feature_map_norm1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%feature_maps/feature_map_norm1/beta/m

9feature_maps/feature_map_norm1/beta/m/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_norm1/beta/m*
_output_shapes
: *
dtype0
²
'feature_maps/feature_map_conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*8
shared_name)'feature_maps/feature_map_conv2/kernel/m
«
;feature_maps/feature_map_conv2/kernel/m/Read/ReadVariableOpReadVariableOp'feature_maps/feature_map_conv2/kernel/m*&
_output_shapes
: @*
dtype0
¢
%feature_maps/feature_map_conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%feature_maps/feature_map_conv2/bias/m

9feature_maps/feature_map_conv2/bias/m/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv2/bias/m*
_output_shapes
:@*
dtype0
¤
&feature_maps/feature_map_norm2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&feature_maps/feature_map_norm2/gamma/m

:feature_maps/feature_map_norm2/gamma/m/Read/ReadVariableOpReadVariableOp&feature_maps/feature_map_norm2/gamma/m*
_output_shapes
:@*
dtype0
¢
%feature_maps/feature_map_norm2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%feature_maps/feature_map_norm2/beta/m

9feature_maps/feature_map_norm2/beta/m/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_norm2/beta/m*
_output_shapes
:@*
dtype0
²
'feature_maps/feature_map_conv3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*8
shared_name)'feature_maps/feature_map_conv3/kernel/m
«
;feature_maps/feature_map_conv3/kernel/m/Read/ReadVariableOpReadVariableOp'feature_maps/feature_map_conv3/kernel/m*&
_output_shapes
:@@*
dtype0
¢
%feature_maps/feature_map_conv3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%feature_maps/feature_map_conv3/bias/m

9feature_maps/feature_map_conv3/bias/m/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv3/bias/m*
_output_shapes
:@*
dtype0
¤
&feature_maps/feature_map_norm3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&feature_maps/feature_map_norm3/gamma/m

:feature_maps/feature_map_norm3/gamma/m/Read/ReadVariableOpReadVariableOp&feature_maps/feature_map_norm3/gamma/m*
_output_shapes
:@*
dtype0
¢
%feature_maps/feature_map_norm3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%feature_maps/feature_map_norm3/beta/m

9feature_maps/feature_map_norm3/beta/m/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_norm3/beta/m*
_output_shapes
:@*
dtype0
³
'feature_maps/feature_map_conv4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'feature_maps/feature_map_conv4/kernel/m
¬
;feature_maps/feature_map_conv4/kernel/m/Read/ReadVariableOpReadVariableOp'feature_maps/feature_map_conv4/kernel/m*'
_output_shapes
:@*
dtype0
£
%feature_maps/feature_map_conv4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%feature_maps/feature_map_conv4/bias/m

9feature_maps/feature_map_conv4/bias/m/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv4/bias/m*
_output_shapes	
:*
dtype0
¥
&feature_maps/feature_map_norm4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&feature_maps/feature_map_norm4/gamma/m

:feature_maps/feature_map_norm4/gamma/m/Read/ReadVariableOpReadVariableOp&feature_maps/feature_map_norm4/gamma/m*
_output_shapes	
:*
dtype0
£
%feature_maps/feature_map_norm4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%feature_maps/feature_map_norm4/beta/m

9feature_maps/feature_map_norm4/beta/m/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_norm4/beta/m*
_output_shapes	
:*
dtype0
³
'primary_caps/primary_cap_dconv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*8
shared_name)'primary_caps/primary_cap_dconv/kernel/m
¬
;primary_caps/primary_cap_dconv/kernel/m/Read/ReadVariableOpReadVariableOp'primary_caps/primary_cap_dconv/kernel/m*'
_output_shapes
:		*
dtype0
£
%primary_caps/primary_cap_dconv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%primary_caps/primary_cap_dconv/bias/m

9primary_caps/primary_cap_dconv/bias/m/Read/ReadVariableOpReadVariableOp%primary_caps/primary_cap_dconv/bias/m*
_output_shapes	
:*
dtype0
´
(digit_caps/digit_caps_transform_tensor/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(digit_caps/digit_caps_transform_tensor/v
­
<digit_caps/digit_caps_transform_tensor/v/Read/ReadVariableOpReadVariableOp(digit_caps/digit_caps_transform_tensor/v*&
_output_shapes
:
*
dtype0
¤
"digit_caps/digit_caps_log_priors/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"digit_caps/digit_caps_log_priors/v

6digit_caps/digit_caps_log_priors/v/Read/ReadVariableOpReadVariableOp"digit_caps/digit_caps_log_priors/v*"
_output_shapes
:
*
dtype0
²
'feature_maps/feature_map_conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'feature_maps/feature_map_conv1/kernel/v
«
;feature_maps/feature_map_conv1/kernel/v/Read/ReadVariableOpReadVariableOp'feature_maps/feature_map_conv1/kernel/v*&
_output_shapes
: *
dtype0
¢
%feature_maps/feature_map_conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%feature_maps/feature_map_conv1/bias/v

9feature_maps/feature_map_conv1/bias/v/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv1/bias/v*
_output_shapes
: *
dtype0
¤
&feature_maps/feature_map_norm1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&feature_maps/feature_map_norm1/gamma/v

:feature_maps/feature_map_norm1/gamma/v/Read/ReadVariableOpReadVariableOp&feature_maps/feature_map_norm1/gamma/v*
_output_shapes
: *
dtype0
¢
%feature_maps/feature_map_norm1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%feature_maps/feature_map_norm1/beta/v

9feature_maps/feature_map_norm1/beta/v/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_norm1/beta/v*
_output_shapes
: *
dtype0
²
'feature_maps/feature_map_conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*8
shared_name)'feature_maps/feature_map_conv2/kernel/v
«
;feature_maps/feature_map_conv2/kernel/v/Read/ReadVariableOpReadVariableOp'feature_maps/feature_map_conv2/kernel/v*&
_output_shapes
: @*
dtype0
¢
%feature_maps/feature_map_conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%feature_maps/feature_map_conv2/bias/v

9feature_maps/feature_map_conv2/bias/v/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv2/bias/v*
_output_shapes
:@*
dtype0
¤
&feature_maps/feature_map_norm2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&feature_maps/feature_map_norm2/gamma/v

:feature_maps/feature_map_norm2/gamma/v/Read/ReadVariableOpReadVariableOp&feature_maps/feature_map_norm2/gamma/v*
_output_shapes
:@*
dtype0
¢
%feature_maps/feature_map_norm2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%feature_maps/feature_map_norm2/beta/v

9feature_maps/feature_map_norm2/beta/v/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_norm2/beta/v*
_output_shapes
:@*
dtype0
²
'feature_maps/feature_map_conv3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*8
shared_name)'feature_maps/feature_map_conv3/kernel/v
«
;feature_maps/feature_map_conv3/kernel/v/Read/ReadVariableOpReadVariableOp'feature_maps/feature_map_conv3/kernel/v*&
_output_shapes
:@@*
dtype0
¢
%feature_maps/feature_map_conv3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%feature_maps/feature_map_conv3/bias/v

9feature_maps/feature_map_conv3/bias/v/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv3/bias/v*
_output_shapes
:@*
dtype0
¤
&feature_maps/feature_map_norm3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&feature_maps/feature_map_norm3/gamma/v

:feature_maps/feature_map_norm3/gamma/v/Read/ReadVariableOpReadVariableOp&feature_maps/feature_map_norm3/gamma/v*
_output_shapes
:@*
dtype0
¢
%feature_maps/feature_map_norm3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%feature_maps/feature_map_norm3/beta/v

9feature_maps/feature_map_norm3/beta/v/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_norm3/beta/v*
_output_shapes
:@*
dtype0
³
'feature_maps/feature_map_conv4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'feature_maps/feature_map_conv4/kernel/v
¬
;feature_maps/feature_map_conv4/kernel/v/Read/ReadVariableOpReadVariableOp'feature_maps/feature_map_conv4/kernel/v*'
_output_shapes
:@*
dtype0
£
%feature_maps/feature_map_conv4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%feature_maps/feature_map_conv4/bias/v

9feature_maps/feature_map_conv4/bias/v/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv4/bias/v*
_output_shapes	
:*
dtype0
¥
&feature_maps/feature_map_norm4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&feature_maps/feature_map_norm4/gamma/v

:feature_maps/feature_map_norm4/gamma/v/Read/ReadVariableOpReadVariableOp&feature_maps/feature_map_norm4/gamma/v*
_output_shapes	
:*
dtype0
£
%feature_maps/feature_map_norm4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%feature_maps/feature_map_norm4/beta/v

9feature_maps/feature_map_norm4/beta/v/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_norm4/beta/v*
_output_shapes	
:*
dtype0
³
'primary_caps/primary_cap_dconv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*8
shared_name)'primary_caps/primary_cap_dconv/kernel/v
¬
;primary_caps/primary_cap_dconv/kernel/v/Read/ReadVariableOpReadVariableOp'primary_caps/primary_cap_dconv/kernel/v*'
_output_shapes
:		*
dtype0
£
%primary_caps/primary_cap_dconv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%primary_caps/primary_cap_dconv/bias/v

9primary_caps/primary_cap_dconv/bias/v/Read/ReadVariableOpReadVariableOp%primary_caps/primary_cap_dconv/bias/v*
_output_shapes	
:*
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *óµ>

NoOpNoOp
Ì 
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0* 
valueùBõ Bí

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
	optimizer

signatures
#_self_saveable_object_factories
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature*
>
_init_input_shape
#_self_saveable_object_factories* 

	conv1
	norm1
	conv2
	norm2
	conv3
	norm3
	conv4
	norm4
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses*
Ù
	!dconv
"reshape

#squash
#$_self_saveable_object_factories
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses*

+digit_caps_transform_tensor
+W
,digit_caps_log_priors
,B

-squash
#._self_saveable_object_factories
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses*
³
#5_self_saveable_object_factories
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses* 
Ô
<iter

=beta_1

>beta_2
	?decay
@learning_rate+m,mBmCmDmEmFmGmHmImJmKmLmMmNmOmPmQmZm[m+v,vBvCv Dv¡Ev¢Fv£Gv¤Hv¥Iv¦Jv§Kv¨Lv©MvªNv«Ov¬Pv­Qv®Zv¯[v°*

Aserving_default* 
* 
Ú
B0
C1
D2
E3
F4
G5
H6
I7
J8
K9
L10
M11
N12
O13
P14
Q15
R16
S17
T18
U19
V20
W21
X22
Y23
Z24
[25
+26
,27*

B0
C1
D2
E3
F4
G5
H6
I7
J8
K9
L10
M11
N12
O13
P14
Q15
Z16
[17
+18
,19*
* 
°
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ë

Bkernel
Cbias
#a_self_saveable_object_factories
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses*
ú
haxis
	Dgamma
Ebeta
Rmoving_mean
Smoving_variance
#i_self_saveable_object_factories
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses*
Ë

Fkernel
Gbias
#p_self_saveable_object_factories
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses*
ú
waxis
	Hgamma
Ibeta
Tmoving_mean
Umoving_variance
#x_self_saveable_object_factories
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses*
Ñ

Jkernel
Kbias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	axis
	Lgamma
Mbeta
Vmoving_mean
Wmoving_variance
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
Ò

Nkernel
Obias
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	axis
	Pgamma
Qbeta
Xmoving_mean
Ymoving_variance
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
* 
º
B0
C1
D2
E3
F4
G5
H6
I7
J8
K9
L10
M11
N12
O13
P14
Q15
R16
S17
T18
U19
V20
W21
X22
Y23*
z
B0
C1
D2
E3
F4
G5
H6
I7
J8
K9
L10
M11
N12
O13
P14
Q15*
* 

non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*
* 
* 
Ò

Zkernel
[bias
$¢_self_saveable_object_factories
£	variables
¤trainable_variables
¥regularization_losses
¦	keras_api
§__call__
+¨&call_and_return_all_conditional_losses*
º
$©_self_saveable_object_factories
ª	variables
«trainable_variables
¬regularization_losses
­	keras_api
®__call__
+¯&call_and_return_all_conditional_losses* 
º
$°_self_saveable_object_factories
±	variables
²trainable_variables
³regularization_losses
´	keras_api
µ__call__
+¶&call_and_return_all_conditional_losses* 
* 

Z0
[1*

Z0
[1*
* 

·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*
* 
* 

VARIABLE_VALUE&digit_caps/digit_caps_transform_tensorKlayer_with_weights-2/digit_caps_transform_tensor/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE digit_caps/digit_caps_log_priorsElayer_with_weights-2/digit_caps_log_priors/.ATTRIBUTES/VARIABLE_VALUE*
º
$¼_self_saveable_object_factories
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses* 
* 

+0
,1*

+0
,1*
* 

Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 
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
* 
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
R0
S1
T2
U3
V4
W5
X6
Y7*
'
0
1
2
3
4*
* 
* 
* 
* 

B0
C1*

B0
C1*
* 

Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*
* 
* 
* 
* 
 
D0
E1
R2
S3*

D0
E1*
* 

Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*
* 
* 
* 

F0
G1*

F0
G1*
* 

×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses*
* 
* 
* 
* 
 
H0
I1
T2
U3*

H0
I1*
* 

Ünon_trainable_variables
Ýlayers
Þmetrics
 ßlayer_regularization_losses
àlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses*
* 
* 
* 

J0
K1*

J0
K1*
* 

ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
 
L0
M1
V2
W3*

L0
M1*
* 

ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 

N0
O1*

N0
O1*
* 

ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
 
P0
Q1
X2
Y3*

P0
Q1*
* 

ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
<
R0
S1
T2
U3
V4
W5
X6
Y7*
<
0
1
2
3
4
5
6
7*
* 
* 
* 
* 

Z0
[1*

Z0
[1*
* 

õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
£	variables
¤trainable_variables
¥regularization_losses
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
ª	variables
«trainable_variables
¬regularization_losses
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
±	variables
²trainable_variables
³regularization_losses
µ__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses* 
* 
* 
* 

!0
"1
#2*
* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
½	variables
¾trainable_variables
¿regularization_losses
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses* 
* 
* 
* 
	
-0* 
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

R0
S1*
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
T0
U1*
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
V0
W1*
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
X0
Y1*
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
ª£
VARIABLE_VALUE(digit_caps/digit_caps_transform_tensor/mglayer_with_weights-2/digit_caps_transform_tensor/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"digit_caps/digit_caps_log_priors/malayer_with_weights-2/digit_caps_log_priors/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE'feature_maps/feature_map_conv1/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%feature_maps/feature_map_conv1/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&feature_maps/feature_map_norm1/gamma/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%feature_maps/feature_map_norm1/beta/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE'feature_maps/feature_map_conv2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%feature_maps/feature_map_conv2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&feature_maps/feature_map_norm2/gamma/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%feature_maps/feature_map_norm2/beta/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE'feature_maps/feature_map_conv3/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%feature_maps/feature_map_conv3/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE&feature_maps/feature_map_norm3/gamma/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%feature_maps/feature_map_norm3/beta/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE'feature_maps/feature_map_conv4/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%feature_maps/feature_map_conv4/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE&feature_maps/feature_map_norm4/gamma/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%feature_maps/feature_map_norm4/beta/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE'primary_caps/primary_cap_dconv/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%primary_caps/primary_cap_dconv/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ª£
VARIABLE_VALUE(digit_caps/digit_caps_transform_tensor/vglayer_with_weights-2/digit_caps_transform_tensor/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"digit_caps/digit_caps_log_priors/valayer_with_weights-2/digit_caps_log_priors/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE'feature_maps/feature_map_conv1/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%feature_maps/feature_map_conv1/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&feature_maps/feature_map_norm1/gamma/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%feature_maps/feature_map_norm1/beta/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE'feature_maps/feature_map_conv2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%feature_maps/feature_map_conv2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&feature_maps/feature_map_norm2/gamma/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%feature_maps/feature_map_norm2/beta/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE'feature_maps/feature_map_conv3/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%feature_maps/feature_map_conv3/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE&feature_maps/feature_map_norm3/gamma/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%feature_maps/feature_map_norm3/beta/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE'feature_maps/feature_map_conv4/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%feature_maps/feature_map_conv4/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE&feature_maps/feature_map_norm4/gamma/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%feature_maps/feature_map_norm4/beta/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE'primary_caps/primary_cap_dconv/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%primary_caps/primary_cap_dconv/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_imagesPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
Ü
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_images%feature_maps/feature_map_conv1/kernel#feature_maps/feature_map_conv1/bias$feature_maps/feature_map_norm1/gamma#feature_maps/feature_map_norm1/beta*feature_maps/feature_map_norm1/moving_mean.feature_maps/feature_map_norm1/moving_variance%feature_maps/feature_map_conv2/kernel#feature_maps/feature_map_conv2/bias$feature_maps/feature_map_norm2/gamma#feature_maps/feature_map_norm2/beta*feature_maps/feature_map_norm2/moving_mean.feature_maps/feature_map_norm2/moving_variance%feature_maps/feature_map_conv3/kernel#feature_maps/feature_map_conv3/bias$feature_maps/feature_map_norm3/gamma#feature_maps/feature_map_norm3/beta*feature_maps/feature_map_norm3/moving_mean.feature_maps/feature_map_norm3/moving_variance%feature_maps/feature_map_conv4/kernel#feature_maps/feature_map_conv4/bias$feature_maps/feature_map_norm4/gamma#feature_maps/feature_map_norm4/beta*feature_maps/feature_map_norm4/moving_mean.feature_maps/feature_map_norm4/moving_variance%primary_caps/primary_cap_dconv/kernel#primary_caps/primary_cap_dconv/bias&digit_caps/digit_caps_transform_tensorConst digit_caps/digit_caps_log_priors*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_173268
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ø#
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename:digit_caps/digit_caps_transform_tensor/Read/ReadVariableOp4digit_caps/digit_caps_log_priors/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp9feature_maps/feature_map_conv1/kernel/Read/ReadVariableOp7feature_maps/feature_map_conv1/bias/Read/ReadVariableOp8feature_maps/feature_map_norm1/gamma/Read/ReadVariableOp7feature_maps/feature_map_norm1/beta/Read/ReadVariableOp9feature_maps/feature_map_conv2/kernel/Read/ReadVariableOp7feature_maps/feature_map_conv2/bias/Read/ReadVariableOp8feature_maps/feature_map_norm2/gamma/Read/ReadVariableOp7feature_maps/feature_map_norm2/beta/Read/ReadVariableOp9feature_maps/feature_map_conv3/kernel/Read/ReadVariableOp7feature_maps/feature_map_conv3/bias/Read/ReadVariableOp8feature_maps/feature_map_norm3/gamma/Read/ReadVariableOp7feature_maps/feature_map_norm3/beta/Read/ReadVariableOp9feature_maps/feature_map_conv4/kernel/Read/ReadVariableOp7feature_maps/feature_map_conv4/bias/Read/ReadVariableOp8feature_maps/feature_map_norm4/gamma/Read/ReadVariableOp7feature_maps/feature_map_norm4/beta/Read/ReadVariableOp>feature_maps/feature_map_norm1/moving_mean/Read/ReadVariableOpBfeature_maps/feature_map_norm1/moving_variance/Read/ReadVariableOp>feature_maps/feature_map_norm2/moving_mean/Read/ReadVariableOpBfeature_maps/feature_map_norm2/moving_variance/Read/ReadVariableOp>feature_maps/feature_map_norm3/moving_mean/Read/ReadVariableOpBfeature_maps/feature_map_norm3/moving_variance/Read/ReadVariableOp>feature_maps/feature_map_norm4/moving_mean/Read/ReadVariableOpBfeature_maps/feature_map_norm4/moving_variance/Read/ReadVariableOp9primary_caps/primary_cap_dconv/kernel/Read/ReadVariableOp7primary_caps/primary_cap_dconv/bias/Read/ReadVariableOp<digit_caps/digit_caps_transform_tensor/m/Read/ReadVariableOp6digit_caps/digit_caps_log_priors/m/Read/ReadVariableOp;feature_maps/feature_map_conv1/kernel/m/Read/ReadVariableOp9feature_maps/feature_map_conv1/bias/m/Read/ReadVariableOp:feature_maps/feature_map_norm1/gamma/m/Read/ReadVariableOp9feature_maps/feature_map_norm1/beta/m/Read/ReadVariableOp;feature_maps/feature_map_conv2/kernel/m/Read/ReadVariableOp9feature_maps/feature_map_conv2/bias/m/Read/ReadVariableOp:feature_maps/feature_map_norm2/gamma/m/Read/ReadVariableOp9feature_maps/feature_map_norm2/beta/m/Read/ReadVariableOp;feature_maps/feature_map_conv3/kernel/m/Read/ReadVariableOp9feature_maps/feature_map_conv3/bias/m/Read/ReadVariableOp:feature_maps/feature_map_norm3/gamma/m/Read/ReadVariableOp9feature_maps/feature_map_norm3/beta/m/Read/ReadVariableOp;feature_maps/feature_map_conv4/kernel/m/Read/ReadVariableOp9feature_maps/feature_map_conv4/bias/m/Read/ReadVariableOp:feature_maps/feature_map_norm4/gamma/m/Read/ReadVariableOp9feature_maps/feature_map_norm4/beta/m/Read/ReadVariableOp;primary_caps/primary_cap_dconv/kernel/m/Read/ReadVariableOp9primary_caps/primary_cap_dconv/bias/m/Read/ReadVariableOp<digit_caps/digit_caps_transform_tensor/v/Read/ReadVariableOp6digit_caps/digit_caps_log_priors/v/Read/ReadVariableOp;feature_maps/feature_map_conv1/kernel/v/Read/ReadVariableOp9feature_maps/feature_map_conv1/bias/v/Read/ReadVariableOp:feature_maps/feature_map_norm1/gamma/v/Read/ReadVariableOp9feature_maps/feature_map_norm1/beta/v/Read/ReadVariableOp;feature_maps/feature_map_conv2/kernel/v/Read/ReadVariableOp9feature_maps/feature_map_conv2/bias/v/Read/ReadVariableOp:feature_maps/feature_map_norm2/gamma/v/Read/ReadVariableOp9feature_maps/feature_map_norm2/beta/v/Read/ReadVariableOp;feature_maps/feature_map_conv3/kernel/v/Read/ReadVariableOp9feature_maps/feature_map_conv3/bias/v/Read/ReadVariableOp:feature_maps/feature_map_norm3/gamma/v/Read/ReadVariableOp9feature_maps/feature_map_norm3/beta/v/Read/ReadVariableOp;feature_maps/feature_map_conv4/kernel/v/Read/ReadVariableOp9feature_maps/feature_map_conv4/bias/v/Read/ReadVariableOp:feature_maps/feature_map_norm4/gamma/v/Read/ReadVariableOp9feature_maps/feature_map_norm4/beta/v/Read/ReadVariableOp;primary_caps/primary_cap_dconv/kernel/v/Read/ReadVariableOp9primary_caps/primary_cap_dconv/bias/v/Read/ReadVariableOpConst_1*V
TinO
M2K	*
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_174043
½
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename&digit_caps/digit_caps_transform_tensor digit_caps/digit_caps_log_priors	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate%feature_maps/feature_map_conv1/kernel#feature_maps/feature_map_conv1/bias$feature_maps/feature_map_norm1/gamma#feature_maps/feature_map_norm1/beta%feature_maps/feature_map_conv2/kernel#feature_maps/feature_map_conv2/bias$feature_maps/feature_map_norm2/gamma#feature_maps/feature_map_norm2/beta%feature_maps/feature_map_conv3/kernel#feature_maps/feature_map_conv3/bias$feature_maps/feature_map_norm3/gamma#feature_maps/feature_map_norm3/beta%feature_maps/feature_map_conv4/kernel#feature_maps/feature_map_conv4/bias$feature_maps/feature_map_norm4/gamma#feature_maps/feature_map_norm4/beta*feature_maps/feature_map_norm1/moving_mean.feature_maps/feature_map_norm1/moving_variance*feature_maps/feature_map_norm2/moving_mean.feature_maps/feature_map_norm2/moving_variance*feature_maps/feature_map_norm3/moving_mean.feature_maps/feature_map_norm3/moving_variance*feature_maps/feature_map_norm4/moving_mean.feature_maps/feature_map_norm4/moving_variance%primary_caps/primary_cap_dconv/kernel#primary_caps/primary_cap_dconv/bias(digit_caps/digit_caps_transform_tensor/m"digit_caps/digit_caps_log_priors/m'feature_maps/feature_map_conv1/kernel/m%feature_maps/feature_map_conv1/bias/m&feature_maps/feature_map_norm1/gamma/m%feature_maps/feature_map_norm1/beta/m'feature_maps/feature_map_conv2/kernel/m%feature_maps/feature_map_conv2/bias/m&feature_maps/feature_map_norm2/gamma/m%feature_maps/feature_map_norm2/beta/m'feature_maps/feature_map_conv3/kernel/m%feature_maps/feature_map_conv3/bias/m&feature_maps/feature_map_norm3/gamma/m%feature_maps/feature_map_norm3/beta/m'feature_maps/feature_map_conv4/kernel/m%feature_maps/feature_map_conv4/bias/m&feature_maps/feature_map_norm4/gamma/m%feature_maps/feature_map_norm4/beta/m'primary_caps/primary_cap_dconv/kernel/m%primary_caps/primary_cap_dconv/bias/m(digit_caps/digit_caps_transform_tensor/v"digit_caps/digit_caps_log_priors/v'feature_maps/feature_map_conv1/kernel/v%feature_maps/feature_map_conv1/bias/v&feature_maps/feature_map_norm1/gamma/v%feature_maps/feature_map_norm1/beta/v'feature_maps/feature_map_conv2/kernel/v%feature_maps/feature_map_conv2/bias/v&feature_maps/feature_map_norm2/gamma/v%feature_maps/feature_map_norm2/beta/v'feature_maps/feature_map_conv3/kernel/v%feature_maps/feature_map_conv3/bias/v&feature_maps/feature_map_norm3/gamma/v%feature_maps/feature_map_norm3/beta/v'feature_maps/feature_map_conv4/kernel/v%feature_maps/feature_map_conv4/bias/v&feature_maps/feature_map_norm4/gamma/v%feature_maps/feature_map_norm4/beta/v'primary_caps/primary_cap_dconv/kernel/v%primary_caps/primary_cap_dconv/bias/v*U
TinN
L2J*
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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_174272©
æ
À
M__inference_feature_map_norm4_layer_call_and_return_conditional_losses_173800

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_feature_maps_layer_call_and_return_conditional_losses_168984
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
0feature_map_conv4_conv2d_readvariableop_resource:@@
1feature_map_conv4_biasadd_readvariableop_resource:	8
)feature_map_norm4_readvariableop_resource:	:
+feature_map_norm4_readvariableop_1_resource:	I
:feature_map_norm4_fusedbatchnormv3_readvariableop_resource:	K
<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource:	
identity¢(feature_map_conv1/BiasAdd/ReadVariableOp¢'feature_map_conv1/Conv2D/ReadVariableOp¢(feature_map_conv2/BiasAdd/ReadVariableOp¢'feature_map_conv2/Conv2D/ReadVariableOp¢(feature_map_conv3/BiasAdd/ReadVariableOp¢'feature_map_conv3/Conv2D/ReadVariableOp¢(feature_map_conv4/BiasAdd/ReadVariableOp¢'feature_map_conv4/Conv2D/ReadVariableOp¢ feature_map_norm1/AssignNewValue¢"feature_map_norm1/AssignNewValue_1¢1feature_map_norm1/FusedBatchNormV3/ReadVariableOp¢3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1¢ feature_map_norm1/ReadVariableOp¢"feature_map_norm1/ReadVariableOp_1¢ feature_map_norm2/AssignNewValue¢"feature_map_norm2/AssignNewValue_1¢1feature_map_norm2/FusedBatchNormV3/ReadVariableOp¢3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1¢ feature_map_norm2/ReadVariableOp¢"feature_map_norm2/ReadVariableOp_1¢ feature_map_norm3/AssignNewValue¢"feature_map_norm3/AssignNewValue_1¢1feature_map_norm3/FusedBatchNormV3/ReadVariableOp¢3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1¢ feature_map_norm3/ReadVariableOp¢"feature_map_norm3/ReadVariableOp_1¢ feature_map_norm4/AssignNewValue¢"feature_map_norm4/AssignNewValue_1¢1feature_map_norm4/FusedBatchNormV3/ReadVariableOp¢3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1¢ feature_map_norm4/ReadVariableOp¢"feature_map_norm4/ReadVariableOp_1 
'feature_map_conv1/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ä
feature_map_conv1/Conv2DConv2Dinput_images/feature_map_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

(feature_map_conv1/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0³
feature_map_conv1/BiasAddBiasAdd!feature_map_conv1/Conv2D:output:00feature_map_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
feature_map_conv1/ReluRelu"feature_map_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 feature_map_norm1/ReadVariableOpReadVariableOp)feature_map_norm1_readvariableop_resource*
_output_shapes
: *
dtype0
"feature_map_norm1/ReadVariableOp_1ReadVariableOp+feature_map_norm1_readvariableop_1_resource*
_output_shapes
: *
dtype0¨
1feature_map_norm1/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¬
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0¼
"feature_map_norm1/FusedBatchNormV3FusedBatchNormV3$feature_map_conv1/Relu:activations:0(feature_map_norm1/ReadVariableOp:value:0*feature_map_norm1/ReadVariableOp_1:value:09feature_map_norm1/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<ø
 feature_map_norm1/AssignNewValueAssignVariableOp:feature_map_norm1_fusedbatchnormv3_readvariableop_resource/feature_map_norm1/FusedBatchNormV3:batch_mean:02^feature_map_norm1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
"feature_map_norm1/AssignNewValue_1AssignVariableOp<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm1/FusedBatchNormV3:batch_variance:04^feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0 
'feature_map_conv2/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Þ
feature_map_conv2/Conv2DConv2D&feature_map_norm1/FusedBatchNormV3:y:0/feature_map_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

(feature_map_conv2/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
feature_map_conv2/BiasAddBiasAdd!feature_map_conv2/Conv2D:output:00feature_map_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
feature_map_conv2/ReluRelu"feature_map_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 feature_map_norm2/ReadVariableOpReadVariableOp)feature_map_norm2_readvariableop_resource*
_output_shapes
:@*
dtype0
"feature_map_norm2/ReadVariableOp_1ReadVariableOp+feature_map_norm2_readvariableop_1_resource*
_output_shapes
:@*
dtype0¨
1feature_map_norm2/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¬
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0¼
"feature_map_norm2/FusedBatchNormV3FusedBatchNormV3$feature_map_conv2/Relu:activations:0(feature_map_norm2/ReadVariableOp:value:0*feature_map_norm2/ReadVariableOp_1:value:09feature_map_norm2/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<ø
 feature_map_norm2/AssignNewValueAssignVariableOp:feature_map_norm2_fusedbatchnormv3_readvariableop_resource/feature_map_norm2/FusedBatchNormV3:batch_mean:02^feature_map_norm2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
"feature_map_norm2/AssignNewValue_1AssignVariableOp<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm2/FusedBatchNormV3:batch_variance:04^feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0 
'feature_map_conv3/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Þ
feature_map_conv3/Conv2DConv2D&feature_map_norm2/FusedBatchNormV3:y:0/feature_map_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

(feature_map_conv3/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
feature_map_conv3/BiasAddBiasAdd!feature_map_conv3/Conv2D:output:00feature_map_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
feature_map_conv3/ReluRelu"feature_map_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 feature_map_norm3/ReadVariableOpReadVariableOp)feature_map_norm3_readvariableop_resource*
_output_shapes
:@*
dtype0
"feature_map_norm3/ReadVariableOp_1ReadVariableOp+feature_map_norm3_readvariableop_1_resource*
_output_shapes
:@*
dtype0¨
1feature_map_norm3/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¬
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0¼
"feature_map_norm3/FusedBatchNormV3FusedBatchNormV3$feature_map_conv3/Relu:activations:0(feature_map_norm3/ReadVariableOp:value:0*feature_map_norm3/ReadVariableOp_1:value:09feature_map_norm3/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<ø
 feature_map_norm3/AssignNewValueAssignVariableOp:feature_map_norm3_fusedbatchnormv3_readvariableop_resource/feature_map_norm3/FusedBatchNormV3:batch_mean:02^feature_map_norm3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
"feature_map_norm3/AssignNewValue_1AssignVariableOp<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm3/FusedBatchNormV3:batch_variance:04^feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0¡
'feature_map_conv4/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ß
feature_map_conv4/Conv2DConv2D&feature_map_norm3/FusedBatchNormV3:y:0/feature_map_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
paddingVALID*
strides

(feature_map_conv4/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0´
feature_map_conv4/BiasAddBiasAdd!feature_map_conv4/Conv2D:output:00feature_map_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		}
feature_map_conv4/ReluRelu"feature_map_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
 feature_map_norm4/ReadVariableOpReadVariableOp)feature_map_norm4_readvariableop_resource*
_output_shapes	
:*
dtype0
"feature_map_norm4/ReadVariableOp_1ReadVariableOp+feature_map_norm4_readvariableop_1_resource*
_output_shapes	
:*
dtype0©
1feature_map_norm4/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0­
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Á
"feature_map_norm4/FusedBatchNormV3FusedBatchNormV3$feature_map_conv4/Relu:activations:0(feature_map_norm4/ReadVariableOp:value:0*feature_map_norm4/ReadVariableOp_1:value:09feature_map_norm4/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ		:::::*
epsilon%o:*
exponential_avg_factor%
×#<ø
 feature_map_norm4/AssignNewValueAssignVariableOp:feature_map_norm4_fusedbatchnormv3_readvariableop_resource/feature_map_norm4/FusedBatchNormV3:batch_mean:02^feature_map_norm4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
"feature_map_norm4/AssignNewValue_1AssignVariableOp<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm4/FusedBatchNormV3:batch_variance:04^feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
NoOpNoOp)^feature_map_conv1/BiasAdd/ReadVariableOp(^feature_map_conv1/Conv2D/ReadVariableOp)^feature_map_conv2/BiasAdd/ReadVariableOp(^feature_map_conv2/Conv2D/ReadVariableOp)^feature_map_conv3/BiasAdd/ReadVariableOp(^feature_map_conv3/Conv2D/ReadVariableOp)^feature_map_conv4/BiasAdd/ReadVariableOp(^feature_map_conv4/Conv2D/ReadVariableOp!^feature_map_norm1/AssignNewValue#^feature_map_norm1/AssignNewValue_12^feature_map_norm1/FusedBatchNormV3/ReadVariableOp4^feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm1/ReadVariableOp#^feature_map_norm1/ReadVariableOp_1!^feature_map_norm2/AssignNewValue#^feature_map_norm2/AssignNewValue_12^feature_map_norm2/FusedBatchNormV3/ReadVariableOp4^feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm2/ReadVariableOp#^feature_map_norm2/ReadVariableOp_1!^feature_map_norm3/AssignNewValue#^feature_map_norm3/AssignNewValue_12^feature_map_norm3/FusedBatchNormV3/ReadVariableOp4^feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm3/ReadVariableOp#^feature_map_norm3/ReadVariableOp_1!^feature_map_norm4/AssignNewValue#^feature_map_norm4/AssignNewValue_12^feature_map_norm4/FusedBatchNormV3/ReadVariableOp4^feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm4/ReadVariableOp#^feature_map_norm4/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 ~
IdentityIdentity&feature_map_norm4/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 2T
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
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_images

»
)__inference_restored_function_body_172322
primary_caps!
unknown:

	unknown_0
	unknown_1:

identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallprimary_capsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_digit_caps_layer_call_and_return_conditional_losses_170334s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameprimary_caps:

_output_shapes
: 
ª'
Ñ
H__inference_primary_caps_layer_call_and_return_conditional_losses_169902
feature_mapsK
0primary_cap_dconv_conv2d_readvariableop_resource:		@
1primary_cap_dconv_biasadd_readvariableop_resource:	
identity¢(primary_cap_dconv/BiasAdd/ReadVariableOp¢'primary_cap_dconv/Conv2D/ReadVariableOp¡
'primary_cap_dconv/Conv2D/ReadVariableOpReadVariableOp0primary_cap_dconv_conv2d_readvariableop_resource*'
_output_shapes
:		*
dtype0Å
primary_cap_dconv/Conv2DConv2Dfeature_maps/primary_cap_dconv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

(primary_cap_dconv/BiasAdd/ReadVariableOpReadVariableOp1primary_cap_dconv_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0´
primary_cap_dconv/BiasAddBiasAdd!primary_cap_dconv/Conv2D:output:00primary_cap_dconv/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
primary_cap_dconv/ReluRelu"primary_cap_dconv/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
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
valueB:µ
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
ÿÿÿÿÿÿÿÿÿe
#primary_cap_reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :ß
!primary_cap_reshape/Reshape/shapePack*primary_cap_reshape/strided_slice:output:0,primary_cap_reshape/Reshape/shape/1:output:0,primary_cap_reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:®
primary_cap_reshape/ReshapeReshape$primary_cap_dconv/Relu:activations:0*primary_cap_reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
primary_cap_squash/norm/mulMul$primary_cap_reshape/Reshape:output:0$primary_cap_reshape/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-primary_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿÂ
primary_cap_squash/norm/SumSumprimary_cap_squash/norm/mul:z:06primary_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
primary_cap_squash/norm/SqrtSqrt$primary_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
primary_cap_squash/ExpExp primary_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
primary_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
primary_cap_squash/truedivRealDiv%primary_cap_squash/truediv/x:output:0primary_cap_squash/Exp:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
primary_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
primary_cap_squash/subSub!primary_cap_squash/sub/x:output:0primary_cap_squash/truediv:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
primary_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
primary_cap_squash/addAddV2 primary_cap_squash/norm/Sqrt:y:0!primary_cap_squash/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
primary_cap_squash/truediv_1RealDiv$primary_cap_reshape/Reshape:output:0primary_cap_squash/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
primary_cap_squash/mulMulprimary_cap_squash/sub:z:0 primary_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^primary_cap_dconv/BiasAdd/ReadVariableOp(^primary_cap_dconv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 m
IdentityIdentityprimary_cap_squash/mul:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ		: : 2T
(primary_cap_dconv/BiasAdd/ReadVariableOp(primary_cap_dconv/BiasAdd/ReadVariableOp2R
'primary_cap_dconv/Conv2D/ReadVariableOp'primary_cap_dconv/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
&
_user_specified_namefeature_maps
Ø

M__inference_feature_map_norm4_layer_call_and_return_conditional_losses_173696

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù 
¤

M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_172871
input_images-
feature_maps_172808: !
feature_maps_172810: !
feature_maps_172812: !
feature_maps_172814: !
feature_maps_172816: !
feature_maps_172818: -
feature_maps_172820: @!
feature_maps_172822:@!
feature_maps_172824:@!
feature_maps_172826:@!
feature_maps_172828:@!
feature_maps_172830:@-
feature_maps_172832:@@!
feature_maps_172834:@!
feature_maps_172836:@!
feature_maps_172838:@!
feature_maps_172840:@!
feature_maps_172842:@.
feature_maps_172844:@"
feature_maps_172846:	"
feature_maps_172848:	"
feature_maps_172850:	"
feature_maps_172852:	"
feature_maps_172854:	.
primary_caps_172857:		"
primary_caps_172859:	+
digit_caps_172862:

digit_caps_172864'
digit_caps_172866:

identity¢"digit_caps/StatefulPartitionedCall¢$feature_maps/StatefulPartitionedCall¢$primary_caps/StatefulPartitionedCallí
$feature_maps/StatefulPartitionedCallStatefulPartitionedCallinput_imagesfeature_maps_172808feature_maps_172810feature_maps_172812feature_maps_172814feature_maps_172816feature_maps_172818feature_maps_172820feature_maps_172822feature_maps_172824feature_maps_172826feature_maps_172828feature_maps_172830feature_maps_172832feature_maps_172834feature_maps_172836feature_maps_172838feature_maps_172840feature_maps_172842feature_maps_172844feature_maps_172846feature_maps_172848feature_maps_172850feature_maps_172852feature_maps_172854*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172248
$primary_caps/StatefulPartitionedCallStatefulPartitionedCall-feature_maps/StatefulPartitionedCall:output:0primary_caps_172857primary_caps_172859*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172306
"digit_caps/StatefulPartitionedCallStatefulPartitionedCall-primary_caps/StatefulPartitionedCall:output:0digit_caps_172862digit_caps_172864digit_caps_172866*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172322æ
digit_probs/PartitionedCallPartitionedCall+digit_caps/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_digit_probs_layer_call_and_return_conditional_losses_172413s
IdentityIdentity$digit_probs/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
NoOpNoOp#^digit_caps/StatefulPartitionedCall%^feature_maps/StatefulPartitionedCall%^primary_caps/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"digit_caps/StatefulPartitionedCall"digit_caps/StatefulPartitionedCall2L
$feature_maps/StatefulPartitionedCall$feature_maps/StatefulPartitionedCall2L
$primary_caps/StatefulPartitionedCall$primary_caps/StatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_images:

_output_shapes
: 

Æ
(__inference_map_while_body_167275_170242$
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
¢map/while/MatMul/ReadVariableOp
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            ¹
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholderDmap/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*&
_output_shapes
:
*
element_dtype0
map/while/MatMul/ReadVariableOpReadVariableOp*map_while_matmul_readvariableop_resource_0*&
_output_shapes
:
*
dtype0±
map/while/MatMulBatchMatMulV2'map/while/MatMul/ReadVariableOp:value:04map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*&
_output_shapes
:
Ý
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholdermap/while/MatMul:output:0*
_output_shapes
: *
element_dtype0: éèÒéèÒQ
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
: r
map/while/NoOpNoOp ^map/while/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 e
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
: ¥
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^map/while/NoOp*
T0*
_output_shapes
: :éèÒ"1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"V
(map_while_matmul_readvariableop_resource*map_while_matmul_readvariableop_resource_0"¸
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


H__inference_feature_maps_layer_call_and_return_conditional_losses_171172
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
0feature_map_conv4_conv2d_readvariableop_resource:@@
1feature_map_conv4_biasadd_readvariableop_resource:	8
)feature_map_norm4_readvariableop_resource:	:
+feature_map_norm4_readvariableop_1_resource:	I
:feature_map_norm4_fusedbatchnormv3_readvariableop_resource:	K
<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource:	
identity¢(feature_map_conv1/BiasAdd/ReadVariableOp¢'feature_map_conv1/Conv2D/ReadVariableOp¢(feature_map_conv2/BiasAdd/ReadVariableOp¢'feature_map_conv2/Conv2D/ReadVariableOp¢(feature_map_conv3/BiasAdd/ReadVariableOp¢'feature_map_conv3/Conv2D/ReadVariableOp¢(feature_map_conv4/BiasAdd/ReadVariableOp¢'feature_map_conv4/Conv2D/ReadVariableOp¢ feature_map_norm1/AssignNewValue¢"feature_map_norm1/AssignNewValue_1¢1feature_map_norm1/FusedBatchNormV3/ReadVariableOp¢3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1¢ feature_map_norm1/ReadVariableOp¢"feature_map_norm1/ReadVariableOp_1¢ feature_map_norm2/AssignNewValue¢"feature_map_norm2/AssignNewValue_1¢1feature_map_norm2/FusedBatchNormV3/ReadVariableOp¢3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1¢ feature_map_norm2/ReadVariableOp¢"feature_map_norm2/ReadVariableOp_1¢ feature_map_norm3/AssignNewValue¢"feature_map_norm3/AssignNewValue_1¢1feature_map_norm3/FusedBatchNormV3/ReadVariableOp¢3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1¢ feature_map_norm3/ReadVariableOp¢"feature_map_norm3/ReadVariableOp_1¢ feature_map_norm4/AssignNewValue¢"feature_map_norm4/AssignNewValue_1¢1feature_map_norm4/FusedBatchNormV3/ReadVariableOp¢3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1¢ feature_map_norm4/ReadVariableOp¢"feature_map_norm4/ReadVariableOp_1 
'feature_map_conv1/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ä
feature_map_conv1/Conv2DConv2Dinput_images/feature_map_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

(feature_map_conv1/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0³
feature_map_conv1/BiasAddBiasAdd!feature_map_conv1/Conv2D:output:00feature_map_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
feature_map_conv1/ReluRelu"feature_map_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 feature_map_norm1/ReadVariableOpReadVariableOp)feature_map_norm1_readvariableop_resource*
_output_shapes
: *
dtype0
"feature_map_norm1/ReadVariableOp_1ReadVariableOp+feature_map_norm1_readvariableop_1_resource*
_output_shapes
: *
dtype0¨
1feature_map_norm1/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¬
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0¼
"feature_map_norm1/FusedBatchNormV3FusedBatchNormV3$feature_map_conv1/Relu:activations:0(feature_map_norm1/ReadVariableOp:value:0*feature_map_norm1/ReadVariableOp_1:value:09feature_map_norm1/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<ø
 feature_map_norm1/AssignNewValueAssignVariableOp:feature_map_norm1_fusedbatchnormv3_readvariableop_resource/feature_map_norm1/FusedBatchNormV3:batch_mean:02^feature_map_norm1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
"feature_map_norm1/AssignNewValue_1AssignVariableOp<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm1/FusedBatchNormV3:batch_variance:04^feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0 
'feature_map_conv2/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Þ
feature_map_conv2/Conv2DConv2D&feature_map_norm1/FusedBatchNormV3:y:0/feature_map_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

(feature_map_conv2/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
feature_map_conv2/BiasAddBiasAdd!feature_map_conv2/Conv2D:output:00feature_map_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
feature_map_conv2/ReluRelu"feature_map_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 feature_map_norm2/ReadVariableOpReadVariableOp)feature_map_norm2_readvariableop_resource*
_output_shapes
:@*
dtype0
"feature_map_norm2/ReadVariableOp_1ReadVariableOp+feature_map_norm2_readvariableop_1_resource*
_output_shapes
:@*
dtype0¨
1feature_map_norm2/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¬
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0¼
"feature_map_norm2/FusedBatchNormV3FusedBatchNormV3$feature_map_conv2/Relu:activations:0(feature_map_norm2/ReadVariableOp:value:0*feature_map_norm2/ReadVariableOp_1:value:09feature_map_norm2/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<ø
 feature_map_norm2/AssignNewValueAssignVariableOp:feature_map_norm2_fusedbatchnormv3_readvariableop_resource/feature_map_norm2/FusedBatchNormV3:batch_mean:02^feature_map_norm2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
"feature_map_norm2/AssignNewValue_1AssignVariableOp<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm2/FusedBatchNormV3:batch_variance:04^feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0 
'feature_map_conv3/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Þ
feature_map_conv3/Conv2DConv2D&feature_map_norm2/FusedBatchNormV3:y:0/feature_map_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

(feature_map_conv3/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
feature_map_conv3/BiasAddBiasAdd!feature_map_conv3/Conv2D:output:00feature_map_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
feature_map_conv3/ReluRelu"feature_map_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 feature_map_norm3/ReadVariableOpReadVariableOp)feature_map_norm3_readvariableop_resource*
_output_shapes
:@*
dtype0
"feature_map_norm3/ReadVariableOp_1ReadVariableOp+feature_map_norm3_readvariableop_1_resource*
_output_shapes
:@*
dtype0¨
1feature_map_norm3/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¬
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0¼
"feature_map_norm3/FusedBatchNormV3FusedBatchNormV3$feature_map_conv3/Relu:activations:0(feature_map_norm3/ReadVariableOp:value:0*feature_map_norm3/ReadVariableOp_1:value:09feature_map_norm3/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<ø
 feature_map_norm3/AssignNewValueAssignVariableOp:feature_map_norm3_fusedbatchnormv3_readvariableop_resource/feature_map_norm3/FusedBatchNormV3:batch_mean:02^feature_map_norm3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
"feature_map_norm3/AssignNewValue_1AssignVariableOp<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm3/FusedBatchNormV3:batch_variance:04^feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0¡
'feature_map_conv4/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ß
feature_map_conv4/Conv2DConv2D&feature_map_norm3/FusedBatchNormV3:y:0/feature_map_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
paddingVALID*
strides

(feature_map_conv4/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0´
feature_map_conv4/BiasAddBiasAdd!feature_map_conv4/Conv2D:output:00feature_map_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		}
feature_map_conv4/ReluRelu"feature_map_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
 feature_map_norm4/ReadVariableOpReadVariableOp)feature_map_norm4_readvariableop_resource*
_output_shapes	
:*
dtype0
"feature_map_norm4/ReadVariableOp_1ReadVariableOp+feature_map_norm4_readvariableop_1_resource*
_output_shapes	
:*
dtype0©
1feature_map_norm4/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0­
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Á
"feature_map_norm4/FusedBatchNormV3FusedBatchNormV3$feature_map_conv4/Relu:activations:0(feature_map_norm4/ReadVariableOp:value:0*feature_map_norm4/ReadVariableOp_1:value:09feature_map_norm4/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ		:::::*
epsilon%o:*
exponential_avg_factor%
×#<ø
 feature_map_norm4/AssignNewValueAssignVariableOp:feature_map_norm4_fusedbatchnormv3_readvariableop_resource/feature_map_norm4/FusedBatchNormV3:batch_mean:02^feature_map_norm4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
"feature_map_norm4/AssignNewValue_1AssignVariableOp<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm4/FusedBatchNormV3:batch_variance:04^feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
NoOpNoOp)^feature_map_conv1/BiasAdd/ReadVariableOp(^feature_map_conv1/Conv2D/ReadVariableOp)^feature_map_conv2/BiasAdd/ReadVariableOp(^feature_map_conv2/Conv2D/ReadVariableOp)^feature_map_conv3/BiasAdd/ReadVariableOp(^feature_map_conv3/Conv2D/ReadVariableOp)^feature_map_conv4/BiasAdd/ReadVariableOp(^feature_map_conv4/Conv2D/ReadVariableOp!^feature_map_norm1/AssignNewValue#^feature_map_norm1/AssignNewValue_12^feature_map_norm1/FusedBatchNormV3/ReadVariableOp4^feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm1/ReadVariableOp#^feature_map_norm1/ReadVariableOp_1!^feature_map_norm2/AssignNewValue#^feature_map_norm2/AssignNewValue_12^feature_map_norm2/FusedBatchNormV3/ReadVariableOp4^feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm2/ReadVariableOp#^feature_map_norm2/ReadVariableOp_1!^feature_map_norm3/AssignNewValue#^feature_map_norm3/AssignNewValue_12^feature_map_norm3/FusedBatchNormV3/ReadVariableOp4^feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm3/ReadVariableOp#^feature_map_norm3/ReadVariableOp_1!^feature_map_norm4/AssignNewValue#^feature_map_norm4/AssignNewValue_12^feature_map_norm4/FusedBatchNormV3/ReadVariableOp4^feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm4/ReadVariableOp#^feature_map_norm4/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 ~
IdentityIdentity&feature_map_norm4/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 2T
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
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_images
Ó#


M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_173203

inputs-
feature_maps_173136: !
feature_maps_173138: !
feature_maps_173140: !
feature_maps_173142: !
feature_maps_173144: !
feature_maps_173146: -
feature_maps_173148: @!
feature_maps_173150:@!
feature_maps_173152:@!
feature_maps_173154:@!
feature_maps_173156:@!
feature_maps_173158:@-
feature_maps_173160:@@!
feature_maps_173162:@!
feature_maps_173164:@!
feature_maps_173166:@!
feature_maps_173168:@!
feature_maps_173170:@.
feature_maps_173172:@"
feature_maps_173174:	"
feature_maps_173176:	"
feature_maps_173178:	"
feature_maps_173180:	"
feature_maps_173182:	.
primary_caps_173185:		"
primary_caps_173187:	+
digit_caps_173190:

digit_caps_173192'
digit_caps_173194:

identity¢"digit_caps/StatefulPartitionedCall¢$feature_maps/StatefulPartitionedCall¢$primary_caps/StatefulPartitionedCallß
$feature_maps/StatefulPartitionedCallStatefulPartitionedCallinputsfeature_maps_173136feature_maps_173138feature_maps_173140feature_maps_173142feature_maps_173144feature_maps_173146feature_maps_173148feature_maps_173150feature_maps_173152feature_maps_173154feature_maps_173156feature_maps_173158feature_maps_173160feature_maps_173162feature_maps_173164feature_maps_173166feature_maps_173168feature_maps_173170feature_maps_173172feature_maps_173174feature_maps_173176feature_maps_173178feature_maps_173180feature_maps_173182*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172617
$primary_caps/StatefulPartitionedCallStatefulPartitionedCall-feature_maps/StatefulPartitionedCall:output:0primary_caps_173185primary_caps_173187*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172306
"digit_caps/StatefulPartitionedCallStatefulPartitionedCall-primary_caps/StatefulPartitionedCall:output:0digit_caps_173190digit_caps_173192digit_caps_173194*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172322«
digit_probs/norm/mulMul+digit_caps/StatefulPartitionedCall:output:0+digit_caps/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
y
&digit_probs/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ­
digit_probs/norm/SumSumdigit_probs/norm/mul:z:0/digit_probs/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
	keep_dims(r
digit_probs/norm/SqrtSqrtdigit_probs/norm/Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

digit_probs/norm/SqueezeSqueezedigit_probs/norm/Sqrt:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
squeeze_dims

ÿÿÿÿÿÿÿÿÿp
IdentityIdentity!digit_probs/norm/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
NoOpNoOp#^digit_caps/StatefulPartitionedCall%^feature_maps/StatefulPartitionedCall%^primary_caps/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"digit_caps/StatefulPartitionedCall"digit_caps/StatefulPartitionedCall2L
$feature_maps/StatefulPartitionedCall$feature_maps/StatefulPartitionedCall2L
$primary_caps/StatefulPartitionedCall$primary_caps/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
Ö
¼
M__inference_feature_map_norm1_layer_call_and_return_conditional_losses_173349

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Í
ã
(__inference_map_while_cond_167274_168874$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1$
 map_while_less_map_strided_slice<
8map_while_map_while_cond_167274___redundant_placeholder0<
8map_while_map_while_cond_167274___redundant_placeholder1
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
	
Í
2__inference_feature_map_norm1_layer_call_fn_173373

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_feature_map_norm1_layer_call_and_return_conditional_losses_173318
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
®
H
,__inference_digit_probs_layer_call_fn_173278

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_digit_probs_layer_call_and_return_conditional_losses_172494`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ö
¼
M__inference_feature_map_norm3_layer_call_and_return_conditional_losses_173601

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¶
c
G__inference_digit_probs_layer_call_and_return_conditional_losses_172494

inputs
identityU
norm/mulMulinputsinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
m
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
	keep_dims(Z
	norm/SqrtSqrtnorm/Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
x
norm/SqueezeSqueezenorm/Sqrt:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ]
IdentityIdentitynorm/Squeeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Í
ã
(__inference_map_while_cond_165420_168887$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1$
 map_while_less_map_strided_slice<
8map_while_map_while_cond_165420___redundant_placeholder0<
8map_while_map_while_cond_165420___redundant_placeholder1
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
	
Í
2__inference_feature_map_norm1_layer_call_fn_173386

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_feature_map_norm1_layer_call_and_return_conditional_losses_173349
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
â/
¸
!__inference__wrapped_model_172336
input_images?
%efficient_capsnet_feature_maps_172249: 3
%efficient_capsnet_feature_maps_172251: 3
%efficient_capsnet_feature_maps_172253: 3
%efficient_capsnet_feature_maps_172255: 3
%efficient_capsnet_feature_maps_172257: 3
%efficient_capsnet_feature_maps_172259: ?
%efficient_capsnet_feature_maps_172261: @3
%efficient_capsnet_feature_maps_172263:@3
%efficient_capsnet_feature_maps_172265:@3
%efficient_capsnet_feature_maps_172267:@3
%efficient_capsnet_feature_maps_172269:@3
%efficient_capsnet_feature_maps_172271:@?
%efficient_capsnet_feature_maps_172273:@@3
%efficient_capsnet_feature_maps_172275:@3
%efficient_capsnet_feature_maps_172277:@3
%efficient_capsnet_feature_maps_172279:@3
%efficient_capsnet_feature_maps_172281:@3
%efficient_capsnet_feature_maps_172283:@@
%efficient_capsnet_feature_maps_172285:@4
%efficient_capsnet_feature_maps_172287:	4
%efficient_capsnet_feature_maps_172289:	4
%efficient_capsnet_feature_maps_172291:	4
%efficient_capsnet_feature_maps_172293:	4
%efficient_capsnet_feature_maps_172295:	@
%efficient_capsnet_primary_caps_172307:		4
%efficient_capsnet_primary_caps_172309:	=
#efficient_capsnet_digit_caps_172323:
'
#efficient_capsnet_digit_caps_1723259
#efficient_capsnet_digit_caps_172327:

identity¢4Efficient-CapsNet/digit_caps/StatefulPartitionedCall¢6Efficient-CapsNet/feature_maps/StatefulPartitionedCall¢6Efficient-CapsNet/primary_caps/StatefulPartitionedCall¯

6Efficient-CapsNet/feature_maps/StatefulPartitionedCallStatefulPartitionedCallinput_images%efficient_capsnet_feature_maps_172249%efficient_capsnet_feature_maps_172251%efficient_capsnet_feature_maps_172253%efficient_capsnet_feature_maps_172255%efficient_capsnet_feature_maps_172257%efficient_capsnet_feature_maps_172259%efficient_capsnet_feature_maps_172261%efficient_capsnet_feature_maps_172263%efficient_capsnet_feature_maps_172265%efficient_capsnet_feature_maps_172267%efficient_capsnet_feature_maps_172269%efficient_capsnet_feature_maps_172271%efficient_capsnet_feature_maps_172273%efficient_capsnet_feature_maps_172275%efficient_capsnet_feature_maps_172277%efficient_capsnet_feature_maps_172279%efficient_capsnet_feature_maps_172281%efficient_capsnet_feature_maps_172283%efficient_capsnet_feature_maps_172285%efficient_capsnet_feature_maps_172287%efficient_capsnet_feature_maps_172289%efficient_capsnet_feature_maps_172291%efficient_capsnet_feature_maps_172293%efficient_capsnet_feature_maps_172295*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172248×
6Efficient-CapsNet/primary_caps/StatefulPartitionedCallStatefulPartitionedCall?Efficient-CapsNet/feature_maps/StatefulPartitionedCall:output:0%efficient_capsnet_primary_caps_172307%efficient_capsnet_primary_caps_172309*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172306÷
4Efficient-CapsNet/digit_caps/StatefulPartitionedCallStatefulPartitionedCall?Efficient-CapsNet/primary_caps/StatefulPartitionedCall:output:0#efficient_capsnet_digit_caps_172323#efficient_capsnet_digit_caps_172325#efficient_capsnet_digit_caps_172327*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172322á
&Efficient-CapsNet/digit_probs/norm/mulMul=Efficient-CapsNet/digit_caps/StatefulPartitionedCall:output:0=Efficient-CapsNet/digit_caps/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

8Efficient-CapsNet/digit_probs/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿã
&Efficient-CapsNet/digit_probs/norm/SumSum*Efficient-CapsNet/digit_probs/norm/mul:z:0AEfficient-CapsNet/digit_probs/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
	keep_dims(
'Efficient-CapsNet/digit_probs/norm/SqrtSqrt/Efficient-CapsNet/digit_probs/norm/Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
´
*Efficient-CapsNet/digit_probs/norm/SqueezeSqueeze+Efficient-CapsNet/digit_probs/norm/Sqrt:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ
IdentityIdentity3Efficient-CapsNet/digit_probs/norm/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ï
NoOpNoOp5^Efficient-CapsNet/digit_caps/StatefulPartitionedCall7^Efficient-CapsNet/feature_maps/StatefulPartitionedCall7^Efficient-CapsNet/primary_caps/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2l
4Efficient-CapsNet/digit_caps/StatefulPartitionedCall4Efficient-CapsNet/digit_caps/StatefulPartitionedCall2p
6Efficient-CapsNet/feature_maps/StatefulPartitionedCall6Efficient-CapsNet/feature_maps/StatefulPartitionedCall2p
6Efficient-CapsNet/primary_caps/StatefulPartitionedCall6Efficient-CapsNet/primary_caps/StatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_images:

_output_shapes
: 
ª

2__inference_Efficient-CapsNet_layer_call_fn_173000

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

unknown_17:@

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	%

unknown_23:		

unknown_24:	$

unknown_25:


unknown_26 

unknown_27:

identity¢StatefulPartitionedCallÕ
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
:ÿÿÿÿÿÿÿÿÿ
*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_172416o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
¶
½
+__inference_digit_caps_layer_call_fn_169175
primary_caps!
unknown:

	unknown_0
	unknown_1:

identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallprimary_capsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_digit_caps_layer_call_and_return_conditional_losses_169167`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameprimary_caps:

_output_shapes
: 
Ñ
 (
__inference__traced_save_174043
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
>savev2_primary_caps_primary_cap_dconv_bias_read_readvariableopG
Csavev2_digit_caps_digit_caps_transform_tensor_m_read_readvariableopA
=savev2_digit_caps_digit_caps_log_priors_m_read_readvariableopF
Bsavev2_feature_maps_feature_map_conv1_kernel_m_read_readvariableopD
@savev2_feature_maps_feature_map_conv1_bias_m_read_readvariableopE
Asavev2_feature_maps_feature_map_norm1_gamma_m_read_readvariableopD
@savev2_feature_maps_feature_map_norm1_beta_m_read_readvariableopF
Bsavev2_feature_maps_feature_map_conv2_kernel_m_read_readvariableopD
@savev2_feature_maps_feature_map_conv2_bias_m_read_readvariableopE
Asavev2_feature_maps_feature_map_norm2_gamma_m_read_readvariableopD
@savev2_feature_maps_feature_map_norm2_beta_m_read_readvariableopF
Bsavev2_feature_maps_feature_map_conv3_kernel_m_read_readvariableopD
@savev2_feature_maps_feature_map_conv3_bias_m_read_readvariableopE
Asavev2_feature_maps_feature_map_norm3_gamma_m_read_readvariableopD
@savev2_feature_maps_feature_map_norm3_beta_m_read_readvariableopF
Bsavev2_feature_maps_feature_map_conv4_kernel_m_read_readvariableopD
@savev2_feature_maps_feature_map_conv4_bias_m_read_readvariableopE
Asavev2_feature_maps_feature_map_norm4_gamma_m_read_readvariableopD
@savev2_feature_maps_feature_map_norm4_beta_m_read_readvariableopF
Bsavev2_primary_caps_primary_cap_dconv_kernel_m_read_readvariableopD
@savev2_primary_caps_primary_cap_dconv_bias_m_read_readvariableopG
Csavev2_digit_caps_digit_caps_transform_tensor_v_read_readvariableopA
=savev2_digit_caps_digit_caps_log_priors_v_read_readvariableopF
Bsavev2_feature_maps_feature_map_conv1_kernel_v_read_readvariableopD
@savev2_feature_maps_feature_map_conv1_bias_v_read_readvariableopE
Asavev2_feature_maps_feature_map_norm1_gamma_v_read_readvariableopD
@savev2_feature_maps_feature_map_norm1_beta_v_read_readvariableopF
Bsavev2_feature_maps_feature_map_conv2_kernel_v_read_readvariableopD
@savev2_feature_maps_feature_map_conv2_bias_v_read_readvariableopE
Asavev2_feature_maps_feature_map_norm2_gamma_v_read_readvariableopD
@savev2_feature_maps_feature_map_norm2_beta_v_read_readvariableopF
Bsavev2_feature_maps_feature_map_conv3_kernel_v_read_readvariableopD
@savev2_feature_maps_feature_map_conv3_bias_v_read_readvariableopE
Asavev2_feature_maps_feature_map_norm3_gamma_v_read_readvariableopD
@savev2_feature_maps_feature_map_norm3_beta_v_read_readvariableopF
Bsavev2_feature_maps_feature_map_conv4_kernel_v_read_readvariableopD
@savev2_feature_maps_feature_map_conv4_bias_v_read_readvariableopE
Asavev2_feature_maps_feature_map_norm4_gamma_v_read_readvariableopD
@savev2_feature_maps_feature_map_norm4_beta_v_read_readvariableopF
Bsavev2_primary_caps_primary_cap_dconv_kernel_v_read_readvariableopD
@savev2_primary_caps_primary_cap_dconv_bias_v_read_readvariableop
savev2_const_1

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Û"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*"
valueú!B÷!JBKlayer_with_weights-2/digit_caps_transform_tensor/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-2/digit_caps_log_priors/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEBglayer_with_weights-2/digit_caps_transform_tensor/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBalayer_with_weights-2/digit_caps_log_priors/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBglayer_with_weights-2/digit_caps_transform_tensor/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBalayer_with_weights-2/digit_caps_log_priors/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*©
valueBJB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B '
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Asavev2_digit_caps_digit_caps_transform_tensor_read_readvariableop;savev2_digit_caps_digit_caps_log_priors_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop@savev2_feature_maps_feature_map_conv1_kernel_read_readvariableop>savev2_feature_maps_feature_map_conv1_bias_read_readvariableop?savev2_feature_maps_feature_map_norm1_gamma_read_readvariableop>savev2_feature_maps_feature_map_norm1_beta_read_readvariableop@savev2_feature_maps_feature_map_conv2_kernel_read_readvariableop>savev2_feature_maps_feature_map_conv2_bias_read_readvariableop?savev2_feature_maps_feature_map_norm2_gamma_read_readvariableop>savev2_feature_maps_feature_map_norm2_beta_read_readvariableop@savev2_feature_maps_feature_map_conv3_kernel_read_readvariableop>savev2_feature_maps_feature_map_conv3_bias_read_readvariableop?savev2_feature_maps_feature_map_norm3_gamma_read_readvariableop>savev2_feature_maps_feature_map_norm3_beta_read_readvariableop@savev2_feature_maps_feature_map_conv4_kernel_read_readvariableop>savev2_feature_maps_feature_map_conv4_bias_read_readvariableop?savev2_feature_maps_feature_map_norm4_gamma_read_readvariableop>savev2_feature_maps_feature_map_norm4_beta_read_readvariableopEsavev2_feature_maps_feature_map_norm1_moving_mean_read_readvariableopIsavev2_feature_maps_feature_map_norm1_moving_variance_read_readvariableopEsavev2_feature_maps_feature_map_norm2_moving_mean_read_readvariableopIsavev2_feature_maps_feature_map_norm2_moving_variance_read_readvariableopEsavev2_feature_maps_feature_map_norm3_moving_mean_read_readvariableopIsavev2_feature_maps_feature_map_norm3_moving_variance_read_readvariableopEsavev2_feature_maps_feature_map_norm4_moving_mean_read_readvariableopIsavev2_feature_maps_feature_map_norm4_moving_variance_read_readvariableop@savev2_primary_caps_primary_cap_dconv_kernel_read_readvariableop>savev2_primary_caps_primary_cap_dconv_bias_read_readvariableopCsavev2_digit_caps_digit_caps_transform_tensor_m_read_readvariableop=savev2_digit_caps_digit_caps_log_priors_m_read_readvariableopBsavev2_feature_maps_feature_map_conv1_kernel_m_read_readvariableop@savev2_feature_maps_feature_map_conv1_bias_m_read_readvariableopAsavev2_feature_maps_feature_map_norm1_gamma_m_read_readvariableop@savev2_feature_maps_feature_map_norm1_beta_m_read_readvariableopBsavev2_feature_maps_feature_map_conv2_kernel_m_read_readvariableop@savev2_feature_maps_feature_map_conv2_bias_m_read_readvariableopAsavev2_feature_maps_feature_map_norm2_gamma_m_read_readvariableop@savev2_feature_maps_feature_map_norm2_beta_m_read_readvariableopBsavev2_feature_maps_feature_map_conv3_kernel_m_read_readvariableop@savev2_feature_maps_feature_map_conv3_bias_m_read_readvariableopAsavev2_feature_maps_feature_map_norm3_gamma_m_read_readvariableop@savev2_feature_maps_feature_map_norm3_beta_m_read_readvariableopBsavev2_feature_maps_feature_map_conv4_kernel_m_read_readvariableop@savev2_feature_maps_feature_map_conv4_bias_m_read_readvariableopAsavev2_feature_maps_feature_map_norm4_gamma_m_read_readvariableop@savev2_feature_maps_feature_map_norm4_beta_m_read_readvariableopBsavev2_primary_caps_primary_cap_dconv_kernel_m_read_readvariableop@savev2_primary_caps_primary_cap_dconv_bias_m_read_readvariableopCsavev2_digit_caps_digit_caps_transform_tensor_v_read_readvariableop=savev2_digit_caps_digit_caps_log_priors_v_read_readvariableopBsavev2_feature_maps_feature_map_conv1_kernel_v_read_readvariableop@savev2_feature_maps_feature_map_conv1_bias_v_read_readvariableopAsavev2_feature_maps_feature_map_norm1_gamma_v_read_readvariableop@savev2_feature_maps_feature_map_norm1_beta_v_read_readvariableopBsavev2_feature_maps_feature_map_conv2_kernel_v_read_readvariableop@savev2_feature_maps_feature_map_conv2_bias_v_read_readvariableopAsavev2_feature_maps_feature_map_norm2_gamma_v_read_readvariableop@savev2_feature_maps_feature_map_norm2_beta_v_read_readvariableopBsavev2_feature_maps_feature_map_conv3_kernel_v_read_readvariableop@savev2_feature_maps_feature_map_conv3_bias_v_read_readvariableopAsavev2_feature_maps_feature_map_norm3_gamma_v_read_readvariableop@savev2_feature_maps_feature_map_norm3_beta_v_read_readvariableopBsavev2_feature_maps_feature_map_conv4_kernel_v_read_readvariableop@savev2_feature_maps_feature_map_conv4_bias_v_read_readvariableopAsavev2_feature_maps_feature_map_norm4_gamma_v_read_readvariableop@savev2_feature_maps_feature_map_norm4_beta_v_read_readvariableopBsavev2_primary_caps_primary_cap_dconv_kernel_v_read_readvariableop@savev2_primary_caps_primary_cap_dconv_bias_v_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *X
dtypesN
L2J	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*¿
_input_shapes­
ª: :
:
: : : : : : : : : : @:@:@:@:@@:@:@:@:@:::: : :@:@:@:@:::		::
:
: : : : : @:@:@:@:@@:@:@:@:@::::		::
:
: : : : : @:@:@:@:@@:@:@:@:@::::		:: 2(
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
:@:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
:: 
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
::!

_output_shapes	
::- )
'
_output_shapes
:		:!!

_output_shapes	
::,"(
&
_output_shapes
:
:(#$
"
_output_shapes
:
:,$(
&
_output_shapes
: : %

_output_shapes
: : &

_output_shapes
: : '

_output_shapes
: :,((
&
_output_shapes
: @: )

_output_shapes
:@: *

_output_shapes
:@: +

_output_shapes
:@:,,(
&
_output_shapes
:@@: -

_output_shapes
:@: .

_output_shapes
:@: /

_output_shapes
:@:-0)
'
_output_shapes
:@:!1

_output_shapes	
::!2

_output_shapes	
::!3

_output_shapes	
::-4)
'
_output_shapes
:		:!5

_output_shapes	
::,6(
&
_output_shapes
:
:(7$
"
_output_shapes
:
:,8(
&
_output_shapes
: : 9

_output_shapes
: : :

_output_shapes
: : ;

_output_shapes
: :,<(
&
_output_shapes
: @: =

_output_shapes
:@: >

_output_shapes
:@: ?

_output_shapes
:@:,@(
&
_output_shapes
:@@: A

_output_shapes
:@: B

_output_shapes
:@: C

_output_shapes
:@:-D)
'
_output_shapes
:@:!E

_output_shapes	
::!F

_output_shapes	
::!G

_output_shapes	
::-H)
'
_output_shapes
:		:!I

_output_shapes	
::J

_output_shapes
: 
´
¤
2__inference_Efficient-CapsNet_layer_call_fn_172805
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

unknown_17:@

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	%

unknown_23:		

unknown_24:	$

unknown_25:


unknown_26 

unknown_27:

identity¢StatefulPartitionedCallÓ
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
:ÿÿÿÿÿÿÿÿÿ
*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_172681o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_images:

_output_shapes
: 
È

M__inference_feature_map_norm2_layer_call_and_return_conditional_losses_173530

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
îO
û
F__inference_digit_caps_layer_call_and_return_conditional_losses_170334
primary_caps+
map_while_input_6:
	
mul_x3
add_3_readvariableop_resource:

identity¢add_3/ReadVariableOp¢	map/whileP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :y

ExpandDims
ExpandDimsprimary_capsExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
Tile/multiplesConst*
_output_shapes
:*
dtype0*%
valueB"   
         t
TileTileExpandDims:output:0Tile/multiples:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
digit_cap_inputs/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
digit_cap_inputs
ExpandDimsTile:output:0digit_cap_inputs/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
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
valueB:å
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
ÿÿÿÿÿÿÿÿÿÍ
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0: éèÒéèÒ
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensordigit_cap_inputs:output:0Bmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0: éèÒéèÒK
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
ÿÿÿÿÿÿÿÿÿÑ
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0: éèÒéèÒX
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ²
	map/whileWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_1:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0map_while_input_6*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : *#
_read_only_resource_inputs
*
_stateful_parallelism( *4
body,R*
(__inference_map_while_body_167275_170242*4
cond,R*
(__inference_map_while_cond_167274_168874*!
output_shapes
: : : : : : : 
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            Ö
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0«
digit_cap_predictionsSqueeze/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ¬
digit_cap_attentionsBatchMatMulV2digit_cap_predictions:output:0digit_cap_predictions:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
adj_y(j
mulMulmul_xdigit_cap_attentions:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
þÿÿÿÿÿÿÿÿ~
SumSummul:z:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
þÿÿÿÿÿÿÿÿL
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
value	B : ¬
concatConcatV2range:output:0concat/values_1:output:0range_1:output:0concat/values_3:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:o
	transpose	TransposeSum:output:0concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
SoftmaxSoftmaxtranspose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
value	B : ¶
concat_1ConcatV2range_2:output:0concat_1/values_1:output:0range_3:output:0concat_1/values_3:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:
digit_cap_coupling_coefficients	TransposeSoftmax:softmax:0concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
v
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*"
_output_shapes
:
*
dtype0
add_3AddV2#digit_cap_coupling_coefficients:y:0add_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
MatMulBatchMatMulV2	add_3:z:0digit_cap_predictions:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
y
SqueezeSqueezeMatMul:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
squeeze_dims

þÿÿÿÿÿÿÿÿz
digit_cap_squash/norm/mulMulSqueeze:output:0Squeeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
+digit_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ¼
digit_cap_squash/norm/SumSumdigit_cap_squash/norm/mul:z:04digit_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
	keep_dims(|
digit_cap_squash/norm/SqrtSqrt"digit_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
digit_cap_squash/ExpExpdigit_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
digit_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
digit_cap_squash/truedivRealDiv#digit_cap_squash/truediv/x:output:0digit_cap_squash/Exp:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
digit_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
digit_cap_squash/subSubdigit_cap_squash/sub/x:output:0digit_cap_squash/truediv:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
digit_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
digit_cap_squash/addAddV2digit_cap_squash/norm/Sqrt:y:0digit_cap_squash/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

digit_cap_squash/truediv_1RealDivSqueeze:output:0digit_cap_squash/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

digit_cap_squash/mulMuldigit_cap_squash/sub:z:0digit_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
NoOpNoOp^add_3/ReadVariableOp
^map/while*"
_acd_function_control_output(*
_output_shapes
 k
IdentityIdentitydigit_cap_squash/mul:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2,
add_3/ReadVariableOpadd_3/ReadVariableOp2
	map/while	map/while:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameprimary_caps:

_output_shapes
: 
Ïm
ë
H__inference_feature_maps_layer_call_and_return_conditional_losses_168809
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
0feature_map_conv4_conv2d_readvariableop_resource:@@
1feature_map_conv4_biasadd_readvariableop_resource:	8
)feature_map_norm4_readvariableop_resource:	:
+feature_map_norm4_readvariableop_1_resource:	I
:feature_map_norm4_fusedbatchnormv3_readvariableop_resource:	K
<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource:	
identity¢(feature_map_conv1/BiasAdd/ReadVariableOp¢'feature_map_conv1/Conv2D/ReadVariableOp¢(feature_map_conv2/BiasAdd/ReadVariableOp¢'feature_map_conv2/Conv2D/ReadVariableOp¢(feature_map_conv3/BiasAdd/ReadVariableOp¢'feature_map_conv3/Conv2D/ReadVariableOp¢(feature_map_conv4/BiasAdd/ReadVariableOp¢'feature_map_conv4/Conv2D/ReadVariableOp¢1feature_map_norm1/FusedBatchNormV3/ReadVariableOp¢3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1¢ feature_map_norm1/ReadVariableOp¢"feature_map_norm1/ReadVariableOp_1¢1feature_map_norm2/FusedBatchNormV3/ReadVariableOp¢3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1¢ feature_map_norm2/ReadVariableOp¢"feature_map_norm2/ReadVariableOp_1¢1feature_map_norm3/FusedBatchNormV3/ReadVariableOp¢3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1¢ feature_map_norm3/ReadVariableOp¢"feature_map_norm3/ReadVariableOp_1¢1feature_map_norm4/FusedBatchNormV3/ReadVariableOp¢3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1¢ feature_map_norm4/ReadVariableOp¢"feature_map_norm4/ReadVariableOp_1 
'feature_map_conv1/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ä
feature_map_conv1/Conv2DConv2Dinput_images/feature_map_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

(feature_map_conv1/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0³
feature_map_conv1/BiasAddBiasAdd!feature_map_conv1/Conv2D:output:00feature_map_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
feature_map_conv1/ReluRelu"feature_map_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 feature_map_norm1/ReadVariableOpReadVariableOp)feature_map_norm1_readvariableop_resource*
_output_shapes
: *
dtype0
"feature_map_norm1/ReadVariableOp_1ReadVariableOp+feature_map_norm1_readvariableop_1_resource*
_output_shapes
: *
dtype0¨
1feature_map_norm1/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¬
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0®
"feature_map_norm1/FusedBatchNormV3FusedBatchNormV3$feature_map_conv1/Relu:activations:0(feature_map_norm1/ReadVariableOp:value:0*feature_map_norm1/ReadVariableOp_1:value:09feature_map_norm1/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training(  
'feature_map_conv2/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Þ
feature_map_conv2/Conv2DConv2D&feature_map_norm1/FusedBatchNormV3:y:0/feature_map_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

(feature_map_conv2/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
feature_map_conv2/BiasAddBiasAdd!feature_map_conv2/Conv2D:output:00feature_map_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
feature_map_conv2/ReluRelu"feature_map_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 feature_map_norm2/ReadVariableOpReadVariableOp)feature_map_norm2_readvariableop_resource*
_output_shapes
:@*
dtype0
"feature_map_norm2/ReadVariableOp_1ReadVariableOp+feature_map_norm2_readvariableop_1_resource*
_output_shapes
:@*
dtype0¨
1feature_map_norm2/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¬
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0®
"feature_map_norm2/FusedBatchNormV3FusedBatchNormV3$feature_map_conv2/Relu:activations:0(feature_map_norm2/ReadVariableOp:value:0*feature_map_norm2/ReadVariableOp_1:value:09feature_map_norm2/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training(  
'feature_map_conv3/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Þ
feature_map_conv3/Conv2DConv2D&feature_map_norm2/FusedBatchNormV3:y:0/feature_map_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

(feature_map_conv3/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
feature_map_conv3/BiasAddBiasAdd!feature_map_conv3/Conv2D:output:00feature_map_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
feature_map_conv3/ReluRelu"feature_map_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 feature_map_norm3/ReadVariableOpReadVariableOp)feature_map_norm3_readvariableop_resource*
_output_shapes
:@*
dtype0
"feature_map_norm3/ReadVariableOp_1ReadVariableOp+feature_map_norm3_readvariableop_1_resource*
_output_shapes
:@*
dtype0¨
1feature_map_norm3/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¬
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0®
"feature_map_norm3/FusedBatchNormV3FusedBatchNormV3$feature_map_conv3/Relu:activations:0(feature_map_norm3/ReadVariableOp:value:0*feature_map_norm3/ReadVariableOp_1:value:09feature_map_norm3/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( ¡
'feature_map_conv4/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ß
feature_map_conv4/Conv2DConv2D&feature_map_norm3/FusedBatchNormV3:y:0/feature_map_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
paddingVALID*
strides

(feature_map_conv4/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0´
feature_map_conv4/BiasAddBiasAdd!feature_map_conv4/Conv2D:output:00feature_map_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		}
feature_map_conv4/ReluRelu"feature_map_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
 feature_map_norm4/ReadVariableOpReadVariableOp)feature_map_norm4_readvariableop_resource*
_output_shapes	
:*
dtype0
"feature_map_norm4/ReadVariableOp_1ReadVariableOp+feature_map_norm4_readvariableop_1_resource*
_output_shapes	
:*
dtype0©
1feature_map_norm4/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0­
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0³
"feature_map_norm4/FusedBatchNormV3FusedBatchNormV3$feature_map_conv4/Relu:activations:0(feature_map_norm4/ReadVariableOp:value:0*feature_map_norm4/ReadVariableOp_1:value:09feature_map_norm4/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ		:::::*
epsilon%o:*
is_training( â
NoOpNoOp)^feature_map_conv1/BiasAdd/ReadVariableOp(^feature_map_conv1/Conv2D/ReadVariableOp)^feature_map_conv2/BiasAdd/ReadVariableOp(^feature_map_conv2/Conv2D/ReadVariableOp)^feature_map_conv3/BiasAdd/ReadVariableOp(^feature_map_conv3/Conv2D/ReadVariableOp)^feature_map_conv4/BiasAdd/ReadVariableOp(^feature_map_conv4/Conv2D/ReadVariableOp2^feature_map_norm1/FusedBatchNormV3/ReadVariableOp4^feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm1/ReadVariableOp#^feature_map_norm1/ReadVariableOp_12^feature_map_norm2/FusedBatchNormV3/ReadVariableOp4^feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm2/ReadVariableOp#^feature_map_norm2/ReadVariableOp_12^feature_map_norm3/FusedBatchNormV3/ReadVariableOp4^feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm3/ReadVariableOp#^feature_map_norm3/ReadVariableOp_12^feature_map_norm4/FusedBatchNormV3/ReadVariableOp4^feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm4/ReadVariableOp#^feature_map_norm4/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 ~
IdentityIdentity&feature_map_norm4/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 2T
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
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_images
Û#


M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_173133

inputs-
feature_maps_173066: !
feature_maps_173068: !
feature_maps_173070: !
feature_maps_173072: !
feature_maps_173074: !
feature_maps_173076: -
feature_maps_173078: @!
feature_maps_173080:@!
feature_maps_173082:@!
feature_maps_173084:@!
feature_maps_173086:@!
feature_maps_173088:@-
feature_maps_173090:@@!
feature_maps_173092:@!
feature_maps_173094:@!
feature_maps_173096:@!
feature_maps_173098:@!
feature_maps_173100:@.
feature_maps_173102:@"
feature_maps_173104:	"
feature_maps_173106:	"
feature_maps_173108:	"
feature_maps_173110:	"
feature_maps_173112:	.
primary_caps_173115:		"
primary_caps_173117:	+
digit_caps_173120:

digit_caps_173122'
digit_caps_173124:

identity¢"digit_caps/StatefulPartitionedCall¢$feature_maps/StatefulPartitionedCall¢$primary_caps/StatefulPartitionedCallç
$feature_maps/StatefulPartitionedCallStatefulPartitionedCallinputsfeature_maps_173066feature_maps_173068feature_maps_173070feature_maps_173072feature_maps_173074feature_maps_173076feature_maps_173078feature_maps_173080feature_maps_173082feature_maps_173084feature_maps_173086feature_maps_173088feature_maps_173090feature_maps_173092feature_maps_173094feature_maps_173096feature_maps_173098feature_maps_173100feature_maps_173102feature_maps_173104feature_maps_173106feature_maps_173108feature_maps_173110feature_maps_173112*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172248
$primary_caps/StatefulPartitionedCallStatefulPartitionedCall-feature_maps/StatefulPartitionedCall:output:0primary_caps_173115primary_caps_173117*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172306
"digit_caps/StatefulPartitionedCallStatefulPartitionedCall-primary_caps/StatefulPartitionedCall:output:0digit_caps_173120digit_caps_173122digit_caps_173124*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172322«
digit_probs/norm/mulMul+digit_caps/StatefulPartitionedCall:output:0+digit_caps/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
y
&digit_probs/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ­
digit_probs/norm/SumSumdigit_probs/norm/mul:z:0/digit_probs/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
	keep_dims(r
digit_probs/norm/SqrtSqrtdigit_probs/norm/Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

digit_probs/norm/SqueezeSqueezedigit_probs/norm/Sqrt:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
squeeze_dims

ÿÿÿÿÿÿÿÿÿp
IdentityIdentity!digit_probs/norm/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
NoOpNoOp#^digit_caps/StatefulPartitionedCall%^feature_maps/StatefulPartitionedCall%^primary_caps/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"digit_caps/StatefulPartitionedCall"digit_caps/StatefulPartitionedCall2L
$feature_maps/StatefulPartitionedCall$feature_maps/StatefulPartitionedCall2L
$primary_caps/StatefulPartitionedCall$primary_caps/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
ß 


M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_172681

inputs-
feature_maps_172618: !
feature_maps_172620: !
feature_maps_172622: !
feature_maps_172624: !
feature_maps_172626: !
feature_maps_172628: -
feature_maps_172630: @!
feature_maps_172632:@!
feature_maps_172634:@!
feature_maps_172636:@!
feature_maps_172638:@!
feature_maps_172640:@-
feature_maps_172642:@@!
feature_maps_172644:@!
feature_maps_172646:@!
feature_maps_172648:@!
feature_maps_172650:@!
feature_maps_172652:@.
feature_maps_172654:@"
feature_maps_172656:	"
feature_maps_172658:	"
feature_maps_172660:	"
feature_maps_172662:	"
feature_maps_172664:	.
primary_caps_172667:		"
primary_caps_172669:	+
digit_caps_172672:

digit_caps_172674'
digit_caps_172676:

identity¢"digit_caps/StatefulPartitionedCall¢$feature_maps/StatefulPartitionedCall¢$primary_caps/StatefulPartitionedCallß
$feature_maps/StatefulPartitionedCallStatefulPartitionedCallinputsfeature_maps_172618feature_maps_172620feature_maps_172622feature_maps_172624feature_maps_172626feature_maps_172628feature_maps_172630feature_maps_172632feature_maps_172634feature_maps_172636feature_maps_172638feature_maps_172640feature_maps_172642feature_maps_172644feature_maps_172646feature_maps_172648feature_maps_172650feature_maps_172652feature_maps_172654feature_maps_172656feature_maps_172658feature_maps_172660feature_maps_172662feature_maps_172664*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172617
$primary_caps/StatefulPartitionedCallStatefulPartitionedCall-feature_maps/StatefulPartitionedCall:output:0primary_caps_172667primary_caps_172669*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172306
"digit_caps/StatefulPartitionedCallStatefulPartitionedCall-primary_caps/StatefulPartitionedCall:output:0digit_caps_172672digit_caps_172674digit_caps_172676*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172322æ
digit_probs/PartitionedCallPartitionedCall+digit_caps/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_digit_probs_layer_call_and_return_conditional_losses_172494s
IdentityIdentity$digit_probs/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
NoOpNoOp#^digit_caps/StatefulPartitionedCall%^feature_maps/StatefulPartitionedCall%^primary_caps/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"digit_caps/StatefulPartitionedCall"digit_caps/StatefulPartitionedCall2L
$feature_maps/StatefulPartitionedCall$feature_maps/StatefulPartitionedCall2L
$primary_caps/StatefulPartitionedCall$primary_caps/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 


$__inference_signature_wrapper_173268
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

unknown_17:@

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	%

unknown_23:		

unknown_24:	$

unknown_25:


unknown_26 

unknown_27:

identity¢StatefulPartitionedCall¯
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
:ÿÿÿÿÿÿÿÿÿ
*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_172336o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_images:

_output_shapes
: 
È

M__inference_feature_map_norm2_layer_call_and_return_conditional_losses_173444

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ö
¼
M__inference_feature_map_norm2_layer_call_and_return_conditional_losses_173475

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
È

M__inference_feature_map_norm3_layer_call_and_return_conditional_losses_173570

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ïm
ë
H__inference_feature_maps_layer_call_and_return_conditional_losses_169335
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
0feature_map_conv4_conv2d_readvariableop_resource:@@
1feature_map_conv4_biasadd_readvariableop_resource:	8
)feature_map_norm4_readvariableop_resource:	:
+feature_map_norm4_readvariableop_1_resource:	I
:feature_map_norm4_fusedbatchnormv3_readvariableop_resource:	K
<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource:	
identity¢(feature_map_conv1/BiasAdd/ReadVariableOp¢'feature_map_conv1/Conv2D/ReadVariableOp¢(feature_map_conv2/BiasAdd/ReadVariableOp¢'feature_map_conv2/Conv2D/ReadVariableOp¢(feature_map_conv3/BiasAdd/ReadVariableOp¢'feature_map_conv3/Conv2D/ReadVariableOp¢(feature_map_conv4/BiasAdd/ReadVariableOp¢'feature_map_conv4/Conv2D/ReadVariableOp¢1feature_map_norm1/FusedBatchNormV3/ReadVariableOp¢3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1¢ feature_map_norm1/ReadVariableOp¢"feature_map_norm1/ReadVariableOp_1¢1feature_map_norm2/FusedBatchNormV3/ReadVariableOp¢3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1¢ feature_map_norm2/ReadVariableOp¢"feature_map_norm2/ReadVariableOp_1¢1feature_map_norm3/FusedBatchNormV3/ReadVariableOp¢3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1¢ feature_map_norm3/ReadVariableOp¢"feature_map_norm3/ReadVariableOp_1¢1feature_map_norm4/FusedBatchNormV3/ReadVariableOp¢3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1¢ feature_map_norm4/ReadVariableOp¢"feature_map_norm4/ReadVariableOp_1 
'feature_map_conv1/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ä
feature_map_conv1/Conv2DConv2Dinput_images/feature_map_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

(feature_map_conv1/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0³
feature_map_conv1/BiasAddBiasAdd!feature_map_conv1/Conv2D:output:00feature_map_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
feature_map_conv1/ReluRelu"feature_map_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 feature_map_norm1/ReadVariableOpReadVariableOp)feature_map_norm1_readvariableop_resource*
_output_shapes
: *
dtype0
"feature_map_norm1/ReadVariableOp_1ReadVariableOp+feature_map_norm1_readvariableop_1_resource*
_output_shapes
: *
dtype0¨
1feature_map_norm1/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¬
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0®
"feature_map_norm1/FusedBatchNormV3FusedBatchNormV3$feature_map_conv1/Relu:activations:0(feature_map_norm1/ReadVariableOp:value:0*feature_map_norm1/ReadVariableOp_1:value:09feature_map_norm1/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training(  
'feature_map_conv2/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Þ
feature_map_conv2/Conv2DConv2D&feature_map_norm1/FusedBatchNormV3:y:0/feature_map_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

(feature_map_conv2/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
feature_map_conv2/BiasAddBiasAdd!feature_map_conv2/Conv2D:output:00feature_map_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
feature_map_conv2/ReluRelu"feature_map_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 feature_map_norm2/ReadVariableOpReadVariableOp)feature_map_norm2_readvariableop_resource*
_output_shapes
:@*
dtype0
"feature_map_norm2/ReadVariableOp_1ReadVariableOp+feature_map_norm2_readvariableop_1_resource*
_output_shapes
:@*
dtype0¨
1feature_map_norm2/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¬
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0®
"feature_map_norm2/FusedBatchNormV3FusedBatchNormV3$feature_map_conv2/Relu:activations:0(feature_map_norm2/ReadVariableOp:value:0*feature_map_norm2/ReadVariableOp_1:value:09feature_map_norm2/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training(  
'feature_map_conv3/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Þ
feature_map_conv3/Conv2DConv2D&feature_map_norm2/FusedBatchNormV3:y:0/feature_map_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

(feature_map_conv3/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
feature_map_conv3/BiasAddBiasAdd!feature_map_conv3/Conv2D:output:00feature_map_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
feature_map_conv3/ReluRelu"feature_map_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 feature_map_norm3/ReadVariableOpReadVariableOp)feature_map_norm3_readvariableop_resource*
_output_shapes
:@*
dtype0
"feature_map_norm3/ReadVariableOp_1ReadVariableOp+feature_map_norm3_readvariableop_1_resource*
_output_shapes
:@*
dtype0¨
1feature_map_norm3/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¬
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0®
"feature_map_norm3/FusedBatchNormV3FusedBatchNormV3$feature_map_conv3/Relu:activations:0(feature_map_norm3/ReadVariableOp:value:0*feature_map_norm3/ReadVariableOp_1:value:09feature_map_norm3/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( ¡
'feature_map_conv4/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ß
feature_map_conv4/Conv2DConv2D&feature_map_norm3/FusedBatchNormV3:y:0/feature_map_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
paddingVALID*
strides

(feature_map_conv4/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0´
feature_map_conv4/BiasAddBiasAdd!feature_map_conv4/Conv2D:output:00feature_map_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		}
feature_map_conv4/ReluRelu"feature_map_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
 feature_map_norm4/ReadVariableOpReadVariableOp)feature_map_norm4_readvariableop_resource*
_output_shapes	
:*
dtype0
"feature_map_norm4/ReadVariableOp_1ReadVariableOp+feature_map_norm4_readvariableop_1_resource*
_output_shapes	
:*
dtype0©
1feature_map_norm4/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0­
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0³
"feature_map_norm4/FusedBatchNormV3FusedBatchNormV3$feature_map_conv4/Relu:activations:0(feature_map_norm4/ReadVariableOp:value:0*feature_map_norm4/ReadVariableOp_1:value:09feature_map_norm4/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ		:::::*
epsilon%o:*
is_training( â
NoOpNoOp)^feature_map_conv1/BiasAdd/ReadVariableOp(^feature_map_conv1/Conv2D/ReadVariableOp)^feature_map_conv2/BiasAdd/ReadVariableOp(^feature_map_conv2/Conv2D/ReadVariableOp)^feature_map_conv3/BiasAdd/ReadVariableOp(^feature_map_conv3/Conv2D/ReadVariableOp)^feature_map_conv4/BiasAdd/ReadVariableOp(^feature_map_conv4/Conv2D/ReadVariableOp2^feature_map_norm1/FusedBatchNormV3/ReadVariableOp4^feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm1/ReadVariableOp#^feature_map_norm1/ReadVariableOp_12^feature_map_norm2/FusedBatchNormV3/ReadVariableOp4^feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm2/ReadVariableOp#^feature_map_norm2/ReadVariableOp_12^feature_map_norm3/FusedBatchNormV3/ReadVariableOp4^feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm3/ReadVariableOp#^feature_map_norm3/ReadVariableOp_12^feature_map_norm4/FusedBatchNormV3/ReadVariableOp4^feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm4/ReadVariableOp#^feature_map_norm4/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 ~
IdentityIdentity&feature_map_norm4/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 2T
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
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_images
	
Í
2__inference_feature_map_norm3_layer_call_fn_173625

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_feature_map_norm3_layer_call_and_return_conditional_losses_173570
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¶
c
G__inference_digit_probs_layer_call_and_return_conditional_losses_173296

inputs
identityU
norm/mulMulinputsinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
m
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
	keep_dims(Z
	norm/SqrtSqrtnorm/Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
x
norm/SqueezeSqueezenorm/Sqrt:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ]
IdentityIdentitynorm/Squeeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
È

M__inference_feature_map_norm3_layer_call_and_return_conditional_losses_173656

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ö
¼
M__inference_feature_map_norm1_layer_call_and_return_conditional_losses_173422

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¶
c
G__inference_digit_probs_layer_call_and_return_conditional_losses_173287

inputs
identityU
norm/mulMulinputsinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
m
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
	keep_dims(Z
	norm/SqrtSqrtnorm/Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
x
norm/SqueezeSqueezenorm/Sqrt:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ]
IdentityIdentitynorm/Squeeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
	
Í
2__inference_feature_map_norm3_layer_call_fn_173638

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_feature_map_norm3_layer_call_and_return_conditional_losses_173601
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
È

M__inference_feature_map_norm1_layer_call_and_return_conditional_losses_173404

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
æ
À
M__inference_feature_map_norm4_layer_call_and_return_conditional_losses_173727

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ 
¤

M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_172937
input_images-
feature_maps_172874: !
feature_maps_172876: !
feature_maps_172878: !
feature_maps_172880: !
feature_maps_172882: !
feature_maps_172884: -
feature_maps_172886: @!
feature_maps_172888:@!
feature_maps_172890:@!
feature_maps_172892:@!
feature_maps_172894:@!
feature_maps_172896:@-
feature_maps_172898:@@!
feature_maps_172900:@!
feature_maps_172902:@!
feature_maps_172904:@!
feature_maps_172906:@!
feature_maps_172908:@.
feature_maps_172910:@"
feature_maps_172912:	"
feature_maps_172914:	"
feature_maps_172916:	"
feature_maps_172918:	"
feature_maps_172920:	.
primary_caps_172923:		"
primary_caps_172925:	+
digit_caps_172928:

digit_caps_172930'
digit_caps_172932:

identity¢"digit_caps/StatefulPartitionedCall¢$feature_maps/StatefulPartitionedCall¢$primary_caps/StatefulPartitionedCallå
$feature_maps/StatefulPartitionedCallStatefulPartitionedCallinput_imagesfeature_maps_172874feature_maps_172876feature_maps_172878feature_maps_172880feature_maps_172882feature_maps_172884feature_maps_172886feature_maps_172888feature_maps_172890feature_maps_172892feature_maps_172894feature_maps_172896feature_maps_172898feature_maps_172900feature_maps_172902feature_maps_172904feature_maps_172906feature_maps_172908feature_maps_172910feature_maps_172912feature_maps_172914feature_maps_172916feature_maps_172918feature_maps_172920*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172617
$primary_caps/StatefulPartitionedCallStatefulPartitionedCall-feature_maps/StatefulPartitionedCall:output:0primary_caps_172923primary_caps_172925*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172306
"digit_caps/StatefulPartitionedCallStatefulPartitionedCall-primary_caps/StatefulPartitionedCall:output:0digit_caps_172928digit_caps_172930digit_caps_172932*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172322æ
digit_probs/PartitionedCallPartitionedCall+digit_caps/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_digit_probs_layer_call_and_return_conditional_losses_172494s
IdentityIdentity$digit_probs/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
NoOpNoOp#^digit_caps/StatefulPartitionedCall%^feature_maps/StatefulPartitionedCall%^primary_caps/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"digit_caps/StatefulPartitionedCall"digit_caps/StatefulPartitionedCall2L
$feature_maps/StatefulPartitionedCall$feature_maps/StatefulPartitionedCall2L
$primary_caps/StatefulPartitionedCall$primary_caps/StatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_images:

_output_shapes
: 
	
Í
2__inference_feature_map_norm2_layer_call_fn_173499

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_feature_map_norm2_layer_call_and_return_conditional_losses_173444
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¢

2__inference_Efficient-CapsNet_layer_call_fn_173063

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

unknown_17:@

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	%

unknown_23:		

unknown_24:	$

unknown_25:


unknown_26 

unknown_27:

identity¢StatefulPartitionedCallÍ
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
:ÿÿÿÿÿÿÿÿÿ
*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_172681o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
Ø

M__inference_feature_map_norm4_layer_call_and_return_conditional_losses_173782

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


)__inference_restored_function_body_172248
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

unknown_17:@

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	
identity¢StatefulPartitionedCallû
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
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_feature_maps_layer_call_and_return_conditional_losses_169335x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_images
îO
û
F__inference_digit_caps_layer_call_and_return_conditional_losses_169167
primary_caps+
map_while_input_6:
	
mul_x3
add_3_readvariableop_resource:

identity¢add_3/ReadVariableOp¢	map/whileP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :y

ExpandDims
ExpandDimsprimary_capsExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
Tile/multiplesConst*
_output_shapes
:*
dtype0*%
valueB"   
         t
TileTileExpandDims:output:0Tile/multiples:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
digit_cap_inputs/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
digit_cap_inputs
ExpandDimsTile:output:0digit_cap_inputs/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
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
valueB:å
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
ÿÿÿÿÿÿÿÿÿÍ
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0: éèÒéèÒ
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensordigit_cap_inputs:output:0Bmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0: éèÒéèÒK
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
ÿÿÿÿÿÿÿÿÿÑ
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0: éèÒéèÒX
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ²
	map/whileWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_1:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0map_while_input_6*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : *#
_read_only_resource_inputs
*
_stateful_parallelism( *4
body,R*
(__inference_map_while_body_165421_169075*4
cond,R*
(__inference_map_while_cond_165420_168887*!
output_shapes
: : : : : : : 
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            Ö
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0«
digit_cap_predictionsSqueeze/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ¬
digit_cap_attentionsBatchMatMulV2digit_cap_predictions:output:0digit_cap_predictions:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
adj_y(j
mulMulmul_xdigit_cap_attentions:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
þÿÿÿÿÿÿÿÿ~
SumSummul:z:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
þÿÿÿÿÿÿÿÿL
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
value	B : ¬
concatConcatV2range:output:0concat/values_1:output:0range_1:output:0concat/values_3:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:o
	transpose	TransposeSum:output:0concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
SoftmaxSoftmaxtranspose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
value	B : ¶
concat_1ConcatV2range_2:output:0concat_1/values_1:output:0range_3:output:0concat_1/values_3:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:
digit_cap_coupling_coefficients	TransposeSoftmax:softmax:0concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
v
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*"
_output_shapes
:
*
dtype0
add_3AddV2#digit_cap_coupling_coefficients:y:0add_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
MatMulBatchMatMulV2	add_3:z:0digit_cap_predictions:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
y
SqueezeSqueezeMatMul:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
squeeze_dims

þÿÿÿÿÿÿÿÿz
digit_cap_squash/norm/mulMulSqueeze:output:0Squeeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
+digit_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ¼
digit_cap_squash/norm/SumSumdigit_cap_squash/norm/mul:z:04digit_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
	keep_dims(|
digit_cap_squash/norm/SqrtSqrt"digit_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
digit_cap_squash/ExpExpdigit_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
digit_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
digit_cap_squash/truedivRealDiv#digit_cap_squash/truediv/x:output:0digit_cap_squash/Exp:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
digit_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
digit_cap_squash/subSubdigit_cap_squash/sub/x:output:0digit_cap_squash/truediv:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
digit_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
digit_cap_squash/addAddV2digit_cap_squash/norm/Sqrt:y:0digit_cap_squash/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

digit_cap_squash/truediv_1RealDivSqueeze:output:0digit_cap_squash/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

digit_cap_squash/mulMuldigit_cap_squash/sub:z:0digit_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
NoOpNoOp^add_3/ReadVariableOp
^map/while*"
_acd_function_control_output(*
_output_shapes
 k
IdentityIdentitydigit_cap_squash/mul:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2,
add_3/ReadVariableOpadd_3/ReadVariableOp2
	map/while	map/while:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameprimary_caps:

_output_shapes
: 
µ

-__inference_feature_maps_layer_call_fn_169013
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

unknown_17:@

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	
identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ		*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_feature_maps_layer_call_and_return_conditional_losses_168984`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_images
ç 


M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_172416

inputs-
feature_maps_172343: !
feature_maps_172345: !
feature_maps_172347: !
feature_maps_172349: !
feature_maps_172351: !
feature_maps_172353: -
feature_maps_172355: @!
feature_maps_172357:@!
feature_maps_172359:@!
feature_maps_172361:@!
feature_maps_172363:@!
feature_maps_172365:@-
feature_maps_172367:@@!
feature_maps_172369:@!
feature_maps_172371:@!
feature_maps_172373:@!
feature_maps_172375:@!
feature_maps_172377:@.
feature_maps_172379:@"
feature_maps_172381:	"
feature_maps_172383:	"
feature_maps_172385:	"
feature_maps_172387:	"
feature_maps_172389:	.
primary_caps_172392:		"
primary_caps_172394:	+
digit_caps_172397:

digit_caps_172399'
digit_caps_172401:

identity¢"digit_caps/StatefulPartitionedCall¢$feature_maps/StatefulPartitionedCall¢$primary_caps/StatefulPartitionedCallç
$feature_maps/StatefulPartitionedCallStatefulPartitionedCallinputsfeature_maps_172343feature_maps_172345feature_maps_172347feature_maps_172349feature_maps_172351feature_maps_172353feature_maps_172355feature_maps_172357feature_maps_172359feature_maps_172361feature_maps_172363feature_maps_172365feature_maps_172367feature_maps_172369feature_maps_172371feature_maps_172373feature_maps_172375feature_maps_172377feature_maps_172379feature_maps_172381feature_maps_172383feature_maps_172385feature_maps_172387feature_maps_172389*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172248
$primary_caps/StatefulPartitionedCallStatefulPartitionedCall-feature_maps/StatefulPartitionedCall:output:0primary_caps_172392primary_caps_172394*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172306
"digit_caps/StatefulPartitionedCallStatefulPartitionedCall-primary_caps/StatefulPartitionedCall:output:0digit_caps_172397digit_caps_172399digit_caps_172401*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_172322æ
digit_probs/PartitionedCallPartitionedCall+digit_caps/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_digit_probs_layer_call_and_return_conditional_losses_172413s
IdentityIdentity$digit_probs/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
NoOpNoOp#^digit_caps/StatefulPartitionedCall%^feature_maps/StatefulPartitionedCall%^primary_caps/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"digit_caps/StatefulPartitionedCall"digit_caps/StatefulPartitionedCall2L
$feature_maps/StatefulPartitionedCall$feature_maps/StatefulPartitionedCall2L
$primary_caps/StatefulPartitionedCall$primary_caps/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
à
¦
)__inference_restored_function_body_172306
feature_maps"
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallfeature_mapsunknown	unknown_0*
Tin
2*
Tout
2*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_primary_caps_layer_call_and_return_conditional_losses_171030s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ		: : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
&
_user_specified_namefeature_maps
ª'
Ñ
H__inference_primary_caps_layer_call_and_return_conditional_losses_171030
feature_mapsK
0primary_cap_dconv_conv2d_readvariableop_resource:		@
1primary_cap_dconv_biasadd_readvariableop_resource:	
identity¢(primary_cap_dconv/BiasAdd/ReadVariableOp¢'primary_cap_dconv/Conv2D/ReadVariableOp¡
'primary_cap_dconv/Conv2D/ReadVariableOpReadVariableOp0primary_cap_dconv_conv2d_readvariableop_resource*'
_output_shapes
:		*
dtype0Å
primary_cap_dconv/Conv2DConv2Dfeature_maps/primary_cap_dconv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

(primary_cap_dconv/BiasAdd/ReadVariableOpReadVariableOp1primary_cap_dconv_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0´
primary_cap_dconv/BiasAddBiasAdd!primary_cap_dconv/Conv2D:output:00primary_cap_dconv/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
primary_cap_dconv/ReluRelu"primary_cap_dconv/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
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
valueB:µ
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
ÿÿÿÿÿÿÿÿÿe
#primary_cap_reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :ß
!primary_cap_reshape/Reshape/shapePack*primary_cap_reshape/strided_slice:output:0,primary_cap_reshape/Reshape/shape/1:output:0,primary_cap_reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:®
primary_cap_reshape/ReshapeReshape$primary_cap_dconv/Relu:activations:0*primary_cap_reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
primary_cap_squash/norm/mulMul$primary_cap_reshape/Reshape:output:0$primary_cap_reshape/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-primary_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿÂ
primary_cap_squash/norm/SumSumprimary_cap_squash/norm/mul:z:06primary_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
primary_cap_squash/norm/SqrtSqrt$primary_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
primary_cap_squash/ExpExp primary_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
primary_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
primary_cap_squash/truedivRealDiv%primary_cap_squash/truediv/x:output:0primary_cap_squash/Exp:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
primary_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
primary_cap_squash/subSub!primary_cap_squash/sub/x:output:0primary_cap_squash/truediv:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
primary_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
primary_cap_squash/addAddV2 primary_cap_squash/norm/Sqrt:y:0!primary_cap_squash/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
primary_cap_squash/truediv_1RealDiv$primary_cap_reshape/Reshape:output:0primary_cap_squash/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
primary_cap_squash/mulMulprimary_cap_squash/sub:z:0 primary_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^primary_cap_dconv/BiasAdd/ReadVariableOp(^primary_cap_dconv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 m
IdentityIdentityprimary_cap_squash/mul:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ		: : 2T
(primary_cap_dconv/BiasAdd/ReadVariableOp(primary_cap_dconv/BiasAdd/ReadVariableOp2R
'primary_cap_dconv/Conv2D/ReadVariableOp'primary_cap_dconv/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
&
_user_specified_namefeature_maps


)__inference_restored_function_body_172617
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

unknown_17:@

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	
identity¢StatefulPartitionedCalló
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
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_feature_maps_layer_call_and_return_conditional_losses_171172x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_images
	
Ñ
2__inference_feature_map_norm4_layer_call_fn_173751

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_feature_map_norm4_layer_call_and_return_conditional_losses_173696
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

M__inference_feature_map_norm1_layer_call_and_return_conditional_losses_173318

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ö
¼
M__inference_feature_map_norm3_layer_call_and_return_conditional_losses_173674

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
«°
ô7
"__inference__traced_restore_174272
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
9assignvariableop_19_feature_maps_feature_map_conv4_kernel:@F
7assignvariableop_20_feature_maps_feature_map_conv4_bias:	G
8assignvariableop_21_feature_maps_feature_map_norm4_gamma:	F
7assignvariableop_22_feature_maps_feature_map_norm4_beta:	L
>assignvariableop_23_feature_maps_feature_map_norm1_moving_mean: P
Bassignvariableop_24_feature_maps_feature_map_norm1_moving_variance: L
>assignvariableop_25_feature_maps_feature_map_norm2_moving_mean:@P
Bassignvariableop_26_feature_maps_feature_map_norm2_moving_variance:@L
>assignvariableop_27_feature_maps_feature_map_norm3_moving_mean:@P
Bassignvariableop_28_feature_maps_feature_map_norm3_moving_variance:@M
>assignvariableop_29_feature_maps_feature_map_norm4_moving_mean:	Q
Bassignvariableop_30_feature_maps_feature_map_norm4_moving_variance:	T
9assignvariableop_31_primary_caps_primary_cap_dconv_kernel:		F
7assignvariableop_32_primary_caps_primary_cap_dconv_bias:	V
<assignvariableop_33_digit_caps_digit_caps_transform_tensor_m:
L
6assignvariableop_34_digit_caps_digit_caps_log_priors_m:
U
;assignvariableop_35_feature_maps_feature_map_conv1_kernel_m: G
9assignvariableop_36_feature_maps_feature_map_conv1_bias_m: H
:assignvariableop_37_feature_maps_feature_map_norm1_gamma_m: G
9assignvariableop_38_feature_maps_feature_map_norm1_beta_m: U
;assignvariableop_39_feature_maps_feature_map_conv2_kernel_m: @G
9assignvariableop_40_feature_maps_feature_map_conv2_bias_m:@H
:assignvariableop_41_feature_maps_feature_map_norm2_gamma_m:@G
9assignvariableop_42_feature_maps_feature_map_norm2_beta_m:@U
;assignvariableop_43_feature_maps_feature_map_conv3_kernel_m:@@G
9assignvariableop_44_feature_maps_feature_map_conv3_bias_m:@H
:assignvariableop_45_feature_maps_feature_map_norm3_gamma_m:@G
9assignvariableop_46_feature_maps_feature_map_norm3_beta_m:@V
;assignvariableop_47_feature_maps_feature_map_conv4_kernel_m:@H
9assignvariableop_48_feature_maps_feature_map_conv4_bias_m:	I
:assignvariableop_49_feature_maps_feature_map_norm4_gamma_m:	H
9assignvariableop_50_feature_maps_feature_map_norm4_beta_m:	V
;assignvariableop_51_primary_caps_primary_cap_dconv_kernel_m:		H
9assignvariableop_52_primary_caps_primary_cap_dconv_bias_m:	V
<assignvariableop_53_digit_caps_digit_caps_transform_tensor_v:
L
6assignvariableop_54_digit_caps_digit_caps_log_priors_v:
U
;assignvariableop_55_feature_maps_feature_map_conv1_kernel_v: G
9assignvariableop_56_feature_maps_feature_map_conv1_bias_v: H
:assignvariableop_57_feature_maps_feature_map_norm1_gamma_v: G
9assignvariableop_58_feature_maps_feature_map_norm1_beta_v: U
;assignvariableop_59_feature_maps_feature_map_conv2_kernel_v: @G
9assignvariableop_60_feature_maps_feature_map_conv2_bias_v:@H
:assignvariableop_61_feature_maps_feature_map_norm2_gamma_v:@G
9assignvariableop_62_feature_maps_feature_map_norm2_beta_v:@U
;assignvariableop_63_feature_maps_feature_map_conv3_kernel_v:@@G
9assignvariableop_64_feature_maps_feature_map_conv3_bias_v:@H
:assignvariableop_65_feature_maps_feature_map_norm3_gamma_v:@G
9assignvariableop_66_feature_maps_feature_map_norm3_beta_v:@V
;assignvariableop_67_feature_maps_feature_map_conv4_kernel_v:@H
9assignvariableop_68_feature_maps_feature_map_conv4_bias_v:	I
:assignvariableop_69_feature_maps_feature_map_norm4_gamma_v:	H
9assignvariableop_70_feature_maps_feature_map_norm4_beta_v:	V
;assignvariableop_71_primary_caps_primary_cap_dconv_kernel_v:		H
9assignvariableop_72_primary_caps_primary_cap_dconv_bias_v:	
identity_74¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_8¢AssignVariableOp_9Þ"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*"
valueú!B÷!JBKlayer_with_weights-2/digit_caps_transform_tensor/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-2/digit_caps_log_priors/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEBglayer_with_weights-2/digit_caps_transform_tensor/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBalayer_with_weights-2/digit_caps_log_priors/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBglayer_with_weights-2/digit_caps_transform_tensor/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBalayer_with_weights-2/digit_caps_log_priors/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*©
valueBJB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¾
_output_shapes«
¨::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*X
dtypesN
L2J	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOpAssignVariableOp7assignvariableop_digit_caps_digit_caps_transform_tensorIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_1AssignVariableOp3assignvariableop_1_digit_caps_digit_caps_log_priorsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_7AssignVariableOp8assignvariableop_7_feature_maps_feature_map_conv1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_8AssignVariableOp6assignvariableop_8_feature_maps_feature_map_conv1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_9AssignVariableOp7assignvariableop_9_feature_maps_feature_map_norm1_gammaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_10AssignVariableOp7assignvariableop_10_feature_maps_feature_map_norm1_betaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_11AssignVariableOp9assignvariableop_11_feature_maps_feature_map_conv2_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_12AssignVariableOp7assignvariableop_12_feature_maps_feature_map_conv2_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_13AssignVariableOp8assignvariableop_13_feature_maps_feature_map_norm2_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_14AssignVariableOp7assignvariableop_14_feature_maps_feature_map_norm2_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_15AssignVariableOp9assignvariableop_15_feature_maps_feature_map_conv3_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_16AssignVariableOp7assignvariableop_16_feature_maps_feature_map_conv3_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_17AssignVariableOp8assignvariableop_17_feature_maps_feature_map_norm3_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_18AssignVariableOp7assignvariableop_18_feature_maps_feature_map_norm3_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_19AssignVariableOp9assignvariableop_19_feature_maps_feature_map_conv4_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_20AssignVariableOp7assignvariableop_20_feature_maps_feature_map_conv4_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_21AssignVariableOp8assignvariableop_21_feature_maps_feature_map_norm4_gammaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_22AssignVariableOp7assignvariableop_22_feature_maps_feature_map_norm4_betaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_23AssignVariableOp>assignvariableop_23_feature_maps_feature_map_norm1_moving_meanIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_24AssignVariableOpBassignvariableop_24_feature_maps_feature_map_norm1_moving_varianceIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_25AssignVariableOp>assignvariableop_25_feature_maps_feature_map_norm2_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_26AssignVariableOpBassignvariableop_26_feature_maps_feature_map_norm2_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_27AssignVariableOp>assignvariableop_27_feature_maps_feature_map_norm3_moving_meanIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_28AssignVariableOpBassignvariableop_28_feature_maps_feature_map_norm3_moving_varianceIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_29AssignVariableOp>assignvariableop_29_feature_maps_feature_map_norm4_moving_meanIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_30AssignVariableOpBassignvariableop_30_feature_maps_feature_map_norm4_moving_varianceIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_31AssignVariableOp9assignvariableop_31_primary_caps_primary_cap_dconv_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_32AssignVariableOp7assignvariableop_32_primary_caps_primary_cap_dconv_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_33AssignVariableOp<assignvariableop_33_digit_caps_digit_caps_transform_tensor_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_34AssignVariableOp6assignvariableop_34_digit_caps_digit_caps_log_priors_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_35AssignVariableOp;assignvariableop_35_feature_maps_feature_map_conv1_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_36AssignVariableOp9assignvariableop_36_feature_maps_feature_map_conv1_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_37AssignVariableOp:assignvariableop_37_feature_maps_feature_map_norm1_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_38AssignVariableOp9assignvariableop_38_feature_maps_feature_map_norm1_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_39AssignVariableOp;assignvariableop_39_feature_maps_feature_map_conv2_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_40AssignVariableOp9assignvariableop_40_feature_maps_feature_map_conv2_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_41AssignVariableOp:assignvariableop_41_feature_maps_feature_map_norm2_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_42AssignVariableOp9assignvariableop_42_feature_maps_feature_map_norm2_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_43AssignVariableOp;assignvariableop_43_feature_maps_feature_map_conv3_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_44AssignVariableOp9assignvariableop_44_feature_maps_feature_map_conv3_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_45AssignVariableOp:assignvariableop_45_feature_maps_feature_map_norm3_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_46AssignVariableOp9assignvariableop_46_feature_maps_feature_map_norm3_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_47AssignVariableOp;assignvariableop_47_feature_maps_feature_map_conv4_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_48AssignVariableOp9assignvariableop_48_feature_maps_feature_map_conv4_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_49AssignVariableOp:assignvariableop_49_feature_maps_feature_map_norm4_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_50AssignVariableOp9assignvariableop_50_feature_maps_feature_map_norm4_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_51AssignVariableOp;assignvariableop_51_primary_caps_primary_cap_dconv_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_52AssignVariableOp9assignvariableop_52_primary_caps_primary_cap_dconv_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_53AssignVariableOp<assignvariableop_53_digit_caps_digit_caps_transform_tensor_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_54AssignVariableOp6assignvariableop_54_digit_caps_digit_caps_log_priors_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_55AssignVariableOp;assignvariableop_55_feature_maps_feature_map_conv1_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_56AssignVariableOp9assignvariableop_56_feature_maps_feature_map_conv1_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_57AssignVariableOp:assignvariableop_57_feature_maps_feature_map_norm1_gamma_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_58AssignVariableOp9assignvariableop_58_feature_maps_feature_map_norm1_beta_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_59AssignVariableOp;assignvariableop_59_feature_maps_feature_map_conv2_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_60AssignVariableOp9assignvariableop_60_feature_maps_feature_map_conv2_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_61AssignVariableOp:assignvariableop_61_feature_maps_feature_map_norm2_gamma_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_62AssignVariableOp9assignvariableop_62_feature_maps_feature_map_norm2_beta_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_63AssignVariableOp;assignvariableop_63_feature_maps_feature_map_conv3_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_64AssignVariableOp9assignvariableop_64_feature_maps_feature_map_conv3_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_65AssignVariableOp:assignvariableop_65_feature_maps_feature_map_norm3_gamma_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_66AssignVariableOp9assignvariableop_66_feature_maps_feature_map_norm3_beta_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_67AssignVariableOp;assignvariableop_67_feature_maps_feature_map_conv4_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_68AssignVariableOp9assignvariableop_68_feature_maps_feature_map_conv4_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_69AssignVariableOp:assignvariableop_69_feature_maps_feature_map_norm4_gamma_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_70AssignVariableOp9assignvariableop_70_feature_maps_feature_map_norm4_beta_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_71AssignVariableOp;assignvariableop_71_primary_caps_primary_cap_dconv_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_72AssignVariableOp9assignvariableop_72_primary_caps_primary_cap_dconv_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_73Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_74IdentityIdentity_73:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_74Identity_74:output:0*©
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_72AssignVariableOp_722(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

Æ
(__inference_map_while_body_165421_169075$
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
¢map/while/MatMul/ReadVariableOp
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            ¹
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholderDmap/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*&
_output_shapes
:
*
element_dtype0
map/while/MatMul/ReadVariableOpReadVariableOp*map_while_matmul_readvariableop_resource_0*&
_output_shapes
:
*
dtype0±
map/while/MatMulBatchMatMulV2'map/while/MatMul/ReadVariableOp:value:04map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*&
_output_shapes
:
Ý
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholdermap/while/MatMul:output:0*
_output_shapes
: *
element_dtype0: éèÒéèÒQ
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
: r
map/while/NoOpNoOp ^map/while/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 e
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
: ¥
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^map/while/NoOp*
T0*
_output_shapes
: :éèÒ"1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"V
(map_while_matmul_readvariableop_resource*map_while_matmul_readvariableop_resource_0"¸
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
®
H
,__inference_digit_probs_layer_call_fn_173273

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_digit_probs_layer_call_and_return_conditional_losses_172413`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

ª
-__inference_primary_caps_layer_call_fn_170156
feature_maps"
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallfeature_mapsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_primary_caps_layer_call_and_return_conditional_losses_169902`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ		: : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
&
_user_specified_namefeature_maps
	
Í
2__inference_feature_map_norm2_layer_call_fn_173512

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_feature_map_norm2_layer_call_and_return_conditional_losses_173475
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
½

-__inference_feature_maps_layer_call_fn_168838
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

unknown_17:@

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	
identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ		*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_feature_maps_layer_call_and_return_conditional_losses_168809`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_images
	
Ñ
2__inference_feature_map_norm4_layer_call_fn_173764

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_feature_map_norm4_layer_call_and_return_conditional_losses_173727
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶
c
G__inference_digit_probs_layer_call_and_return_conditional_losses_172413

inputs
identityU
norm/mulMulinputsinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
m
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
	keep_dims(Z
	norm/SqrtSqrtnorm/Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
x
norm/SqueezeSqueezenorm/Sqrt:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ]
IdentityIdentitynorm/Squeeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ö
¼
M__inference_feature_map_norm2_layer_call_and_return_conditional_losses_173548

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¼
¤
2__inference_Efficient-CapsNet_layer_call_fn_172477
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

unknown_17:@

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	%

unknown_23:		

unknown_24:	$

unknown_25:


unknown_26 

unknown_27:

identity¢StatefulPartitionedCallÛ
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
:ÿÿÿÿÿÿÿÿÿ
*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_172416o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_images:

_output_shapes
: "ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*À
serving_default¬
M
input_images=
serving_default_input_images:0ÿÿÿÿÿÿÿÿÿ?
digit_probs0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:¤

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
	optimizer

signatures
#_self_saveable_object_factories
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature"
_tf_keras_network
[
_init_input_shape
#_self_saveable_object_factories"
_tf_keras_input_layer
¢
	conv1
	norm1
	conv2
	norm2
	conv3
	norm3
	conv4
	norm4
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_layer
î
	!dconv
"reshape

#squash
#$_self_saveable_object_factories
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_layer
 
+digit_caps_transform_tensor
+W
,digit_caps_log_priors
,B

-squash
#._self_saveable_object_factories
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
Ê
#5_self_saveable_object_factories
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
ã
<iter

=beta_1

>beta_2
	?decay
@learning_rate+m,mBmCmDmEmFmGmHmImJmKmLmMmNmOmPmQmZm[m+v,vBvCv Dv¡Ev¢Fv£Gv¤Hv¥Iv¦Jv§Kv¨Lv©MvªNv«Ov¬Pv­Qv®Zv¯[v°"
	optimizer
,
Aserving_default"
signature_map
 "
trackable_dict_wrapper
ö
B0
C1
D2
E3
F4
G5
H6
I7
J8
K9
L10
M11
N12
O13
P14
Q15
R16
S17
T18
U19
V20
W21
X22
Y23
Z24
[25
+26
,27"
trackable_list_wrapper
¶
B0
C1
D2
E3
F4
G5
H6
I7
J8
K9
L10
M11
N12
O13
P14
Q15
Z16
[17
+18
,19"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
2__inference_Efficient-CapsNet_layer_call_fn_172477
2__inference_Efficient-CapsNet_layer_call_fn_173000
2__inference_Efficient-CapsNet_layer_call_fn_173063
2__inference_Efficient-CapsNet_layer_call_fn_172805À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿ
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_173133
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_173203
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_172871
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_172937À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÑBÎ
!__inference__wrapped_model_172336input_images"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
à

Bkernel
Cbias
#a_self_saveable_object_factories
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses"
_tf_keras_layer

haxis
	Dgamma
Ebeta
Rmoving_mean
Smoving_variance
#i_self_saveable_object_factories
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
à

Fkernel
Gbias
#p_self_saveable_object_factories
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_layer

waxis
	Hgamma
Ibeta
Tmoving_mean
Umoving_variance
#x_self_saveable_object_factories
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
æ

Jkernel
Kbias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer

	axis
	Lgamma
Mbeta
Vmoving_mean
Wmoving_variance
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ç

Nkernel
Obias
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer

	axis
	Pgamma
Qbeta
Xmoving_mean
Ymoving_variance
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_dict_wrapper
Ö
B0
C1
D2
E3
F4
G5
H6
I7
J8
K9
L10
M11
N12
O13
P14
Q15
R16
S17
T18
U19
V20
W21
X22
Y23"
trackable_list_wrapper

B0
C1
D2
E3
F4
G5
H6
I7
J8
K9
L10
M11
N12
O13
P14
Q15"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
2
-__inference_feature_maps_layer_call_fn_168838
-__inference_feature_maps_layer_call_fn_169013¯
¨²¤
FullArgSpec'
args
jinput_images

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
É2Æ
H__inference_feature_maps_layer_call_and_return_conditional_losses_169335
H__inference_feature_maps_layer_call_and_return_conditional_losses_171172¯
¨²¤
FullArgSpec'
args
jinput_images

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ç

Zkernel
[bias
$¢_self_saveable_object_factories
£	variables
¤trainable_variables
¥regularization_losses
¦	keras_api
§__call__
+¨&call_and_return_all_conditional_losses"
_tf_keras_layer
Ñ
$©_self_saveable_object_factories
ª	variables
«trainable_variables
¬regularization_losses
­	keras_api
®__call__
+¯&call_and_return_all_conditional_losses"
_tf_keras_layer
Ñ
$°_self_saveable_object_factories
±	variables
²trainable_variables
³regularization_losses
´	keras_api
µ__call__
+¶&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_dict_wrapper
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
-__inference_primary_caps_layer_call_fn_170156
²
FullArgSpec
args
jfeature_maps
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
H__inference_primary_caps_layer_call_and_return_conditional_losses_171030
²
FullArgSpec
args
jfeature_maps
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
@:>
2&digit_caps/digit_caps_transform_tensor
6:4
2 digit_caps/digit_caps_log_priors
Ñ
$¼_self_saveable_object_factories
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_dict_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
+__inference_digit_caps_layer_call_fn_169175
²
FullArgSpec
args
jprimary_caps
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
F__inference_digit_caps_layer_call_and_return_conditional_losses_170334
²
FullArgSpec
args
jprimary_caps
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
¢2
,__inference_digit_probs_layer_call_fn_173273
,__inference_digit_probs_layer_call_fn_173278À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
G__inference_digit_probs_layer_call_and_return_conditional_losses_173287
G__inference_digit_probs_layer_call_and_return_conditional_losses_173296À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÐBÍ
$__inference_signature_wrapper_173268input_images"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
@:>@2%feature_maps/feature_map_conv4/kernel
2:02#feature_maps/feature_map_conv4/bias
3:12$feature_maps/feature_map_norm4/gamma
2:02#feature_maps/feature_map_norm4/beta
::8  (2*feature_maps/feature_map_norm1/moving_mean
>:<  (2.feature_maps/feature_map_norm1/moving_variance
::8@ (2*feature_maps/feature_map_norm2/moving_mean
>:<@ (2.feature_maps/feature_map_norm2/moving_variance
::8@ (2*feature_maps/feature_map_norm3/moving_mean
>:<@ (2.feature_maps/feature_map_norm3/moving_variance
;:9 (2*feature_maps/feature_map_norm4/moving_mean
?:= (2.feature_maps/feature_map_norm4/moving_variance
@:>		2%primary_caps/primary_cap_dconv/kernel
2:02#primary_caps/primary_cap_dconv/bias
X
R0
S1
T2
U3
V4
W5
X6
Y7"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
D0
E1
R2
S3"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
¢2
2__inference_feature_map_norm1_layer_call_fn_173373
2__inference_feature_map_norm1_layer_call_fn_173386´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
M__inference_feature_map_norm1_layer_call_and_return_conditional_losses_173404
M__inference_feature_map_norm1_layer_call_and_return_conditional_losses_173422´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_dict_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
H0
I1
T2
U3"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ünon_trainable_variables
Ýlayers
Þmetrics
 ßlayer_regularization_losses
àlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
¢2
2__inference_feature_map_norm2_layer_call_fn_173499
2__inference_feature_map_norm2_layer_call_fn_173512´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
M__inference_feature_map_norm2_layer_call_and_return_conditional_losses_173530
M__inference_feature_map_norm2_layer_call_and_return_conditional_losses_173548´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_dict_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
L0
M1
V2
W3"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¢2
2__inference_feature_map_norm3_layer_call_fn_173625
2__inference_feature_map_norm3_layer_call_fn_173638´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
M__inference_feature_map_norm3_layer_call_and_return_conditional_losses_173656
M__inference_feature_map_norm3_layer_call_and_return_conditional_losses_173674´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_dict_wrapper
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
P0
Q1
X2
Y3"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¢2
2__inference_feature_map_norm4_layer_call_fn_173751
2__inference_feature_map_norm4_layer_call_fn_173764´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
M__inference_feature_map_norm4_layer_call_and_return_conditional_losses_173782
M__inference_feature_map_norm4_layer_call_and_return_conditional_losses_173800´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
X
R0
S1
T2
U3
V4
W5
X6
Y7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
£	variables
¤trainable_variables
¥regularization_losses
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
ª	variables
«trainable_variables
¬regularization_losses
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
±	variables
²trainable_variables
³regularization_losses
µ__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
¤2¡
²
FullArgSpec
args
jinput_vector
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¤2¡
²
FullArgSpec
args
jinput_vector
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
½	variables
¾trainable_variables
¿regularization_losses
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
¤2¡
²
FullArgSpec
args
jinput_vector
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¤2¡
²
FullArgSpec
args
jinput_vector
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
'
-0"
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
.
R0
S1"
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
T0
U1"
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
V0
W1"
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
X0
Y1"
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
@:>
2(digit_caps/digit_caps_transform_tensor/m
6:4
2"digit_caps/digit_caps_log_priors/m
?:= 2'feature_maps/feature_map_conv1/kernel/m
1:/ 2%feature_maps/feature_map_conv1/bias/m
2:0 2&feature_maps/feature_map_norm1/gamma/m
1:/ 2%feature_maps/feature_map_norm1/beta/m
?:= @2'feature_maps/feature_map_conv2/kernel/m
1:/@2%feature_maps/feature_map_conv2/bias/m
2:0@2&feature_maps/feature_map_norm2/gamma/m
1:/@2%feature_maps/feature_map_norm2/beta/m
?:=@@2'feature_maps/feature_map_conv3/kernel/m
1:/@2%feature_maps/feature_map_conv3/bias/m
2:0@2&feature_maps/feature_map_norm3/gamma/m
1:/@2%feature_maps/feature_map_norm3/beta/m
@:>@2'feature_maps/feature_map_conv4/kernel/m
2:02%feature_maps/feature_map_conv4/bias/m
3:12&feature_maps/feature_map_norm4/gamma/m
2:02%feature_maps/feature_map_norm4/beta/m
@:>		2'primary_caps/primary_cap_dconv/kernel/m
2:02%primary_caps/primary_cap_dconv/bias/m
@:>
2(digit_caps/digit_caps_transform_tensor/v
6:4
2"digit_caps/digit_caps_log_priors/v
?:= 2'feature_maps/feature_map_conv1/kernel/v
1:/ 2%feature_maps/feature_map_conv1/bias/v
2:0 2&feature_maps/feature_map_norm1/gamma/v
1:/ 2%feature_maps/feature_map_norm1/beta/v
?:= @2'feature_maps/feature_map_conv2/kernel/v
1:/@2%feature_maps/feature_map_conv2/bias/v
2:0@2&feature_maps/feature_map_norm2/gamma/v
1:/@2%feature_maps/feature_map_norm2/beta/v
?:=@@2'feature_maps/feature_map_conv3/kernel/v
1:/@2%feature_maps/feature_map_conv3/bias/v
2:0@2&feature_maps/feature_map_norm3/gamma/v
1:/@2%feature_maps/feature_map_norm3/beta/v
@:>@2'feature_maps/feature_map_conv4/kernel/v
2:02%feature_maps/feature_map_conv4/bias/v
3:12&feature_maps/feature_map_norm4/gamma/v
2:02%feature_maps/feature_map_norm4/beta/v
@:>		2'primary_caps/primary_cap_dconv/kernel/v
2:02%primary_caps/primary_cap_dconv/bias/v
	J
Constà
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_172871BCDERSFGHITUJKLMVWNOPQXYZ[+±,E¢B
;¢8
.+
input_imagesÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 à
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_172937BCDERSFGHITUJKLMVWNOPQXYZ[+±,E¢B
;¢8
.+
input_imagesÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ú
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_173133BCDERSFGHITUJKLMVWNOPQXYZ[+±,?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ú
M__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_173203BCDERSFGHITUJKLMVWNOPQXYZ[+±,?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¸
2__inference_Efficient-CapsNet_layer_call_fn_172477BCDERSFGHITUJKLMVWNOPQXYZ[+±,E¢B
;¢8
.+
input_imagesÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
¸
2__inference_Efficient-CapsNet_layer_call_fn_172805BCDERSFGHITUJKLMVWNOPQXYZ[+±,E¢B
;¢8
.+
input_imagesÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
±
2__inference_Efficient-CapsNet_layer_call_fn_173000{BCDERSFGHITUJKLMVWNOPQXYZ[+±,?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
±
2__inference_Efficient-CapsNet_layer_call_fn_173063{BCDERSFGHITUJKLMVWNOPQXYZ[+±,?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
À
!__inference__wrapped_model_172336BCDERSFGHITUJKLMVWNOPQXYZ[+±,=¢:
3¢0
.+
input_imagesÿÿÿÿÿÿÿÿÿ
ª "9ª6
4
digit_probs%"
digit_probsÿÿÿÿÿÿÿÿÿ
¶
F__inference_digit_caps_layer_call_and_return_conditional_losses_170334l+±,9¢6
/¢,
*'
primary_capsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ

 
+__inference_digit_caps_layer_call_fn_169175_+±,9¢6
/¢,
*'
primary_capsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
¯
G__inference_digit_probs_layer_call_and_return_conditional_losses_173287d;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ


 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¯
G__inference_digit_probs_layer_call_and_return_conditional_losses_173296d;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ


 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
,__inference_digit_probs_layer_call_fn_173273W;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ


 
p 
ª "ÿÿÿÿÿÿÿÿÿ

,__inference_digit_probs_layer_call_fn_173278W;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ


 
p
ª "ÿÿÿÿÿÿÿÿÿ
è
M__inference_feature_map_norm1_layer_call_and_return_conditional_losses_173404DERSM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 è
M__inference_feature_map_norm1_layer_call_and_return_conditional_losses_173422DERSM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 À
2__inference_feature_map_norm1_layer_call_fn_173373DERSM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ À
2__inference_feature_map_norm1_layer_call_fn_173386DERSM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ è
M__inference_feature_map_norm2_layer_call_and_return_conditional_losses_173530HITUM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 è
M__inference_feature_map_norm2_layer_call_and_return_conditional_losses_173548HITUM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 À
2__inference_feature_map_norm2_layer_call_fn_173499HITUM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@À
2__inference_feature_map_norm2_layer_call_fn_173512HITUM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@è
M__inference_feature_map_norm3_layer_call_and_return_conditional_losses_173656LMVWM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 è
M__inference_feature_map_norm3_layer_call_and_return_conditional_losses_173674LMVWM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 À
2__inference_feature_map_norm3_layer_call_fn_173625LMVWM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@À
2__inference_feature_map_norm3_layer_call_fn_173638LMVWM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ê
M__inference_feature_map_norm4_layer_call_and_return_conditional_losses_173782PQXYN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ê
M__inference_feature_map_norm4_layer_call_and_return_conditional_losses_173800PQXYN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
2__inference_feature_map_norm4_layer_call_fn_173751PQXYN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÂ
2__inference_feature_map_norm4_layer_call_fn_173764PQXYN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
H__inference_feature_maps_layer_call_and_return_conditional_losses_169335BCDERSFGHITUJKLMVWNOPQXYA¢>
7¢4
.+
input_imagesÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ		
 Ú
H__inference_feature_maps_layer_call_and_return_conditional_losses_171172BCDERSFGHITUJKLMVWNOPQXYA¢>
7¢4
.+
input_imagesÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ		
 ²
-__inference_feature_maps_layer_call_fn_168838BCDERSFGHITUJKLMVWNOPQXYA¢>
7¢4
.+
input_imagesÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿ		²
-__inference_feature_maps_layer_call_fn_169013BCDERSFGHITUJKLMVWNOPQXYA¢>
7¢4
.+
input_imagesÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ		»
H__inference_primary_caps_layer_call_and_return_conditional_losses_171030oZ[>¢;
4¢1
/,
feature_mapsÿÿÿÿÿÿÿÿÿ		
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_primary_caps_layer_call_fn_170156bZ[>¢;
4¢1
/,
feature_mapsÿÿÿÿÿÿÿÿÿ		
ª "ÿÿÿÿÿÿÿÿÿÓ
$__inference_signature_wrapper_173268ªBCDERSFGHITUJKLMVWNOPQXYZ[+±,M¢J
¢ 
Cª@
>
input_images.+
input_imagesÿÿÿÿÿÿÿÿÿ"9ª6
4
digit_probs%"
digit_probsÿÿÿÿÿÿÿÿÿ
