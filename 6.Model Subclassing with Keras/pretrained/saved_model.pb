Þ¥$
æ
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
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
û
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
2"
Utype:
2"
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
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
H
ShardedFilename
basename	
shard

num_shards
filename
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018¬Þ

!Adam/res_block_1/conv2d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/res_block_1/conv2d_10/bias/v

5Adam/res_block_1/conv2d_10/bias/v/Read/ReadVariableOpReadVariableOp!Adam/res_block_1/conv2d_10/bias/v*
_output_shapes	
:*
dtype0
«
#Adam/res_block_1/conv2d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/res_block_1/conv2d_10/kernel/v
¤
7Adam/res_block_1/conv2d_10/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/res_block_1/conv2d_10/kernel/v*'
_output_shapes
:@*
dtype0
Ë
9Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/v
Ä
MAdam/res_block_1/cnn_block_8/batch_normalization_8/beta/v/Read/ReadVariableOpReadVariableOp9Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/v*
_output_shapes	
:*
dtype0
Í
:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/v
Æ
NAdam/res_block_1/cnn_block_8/batch_normalization_8/gamma/v/Read/ReadVariableOpReadVariableOp:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/v*
_output_shapes	
:*
dtype0
±
,Adam/res_block_1/cnn_block_8/conv2d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/res_block_1/cnn_block_8/conv2d_9/bias/v
ª
@Adam/res_block_1/cnn_block_8/conv2d_9/bias/v/Read/ReadVariableOpReadVariableOp,Adam/res_block_1/cnn_block_8/conv2d_9/bias/v*
_output_shapes	
:*
dtype0
Â
.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/v
»
BAdam/res_block_1/cnn_block_8/conv2d_9/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/v*(
_output_shapes
:*
dtype0
Ë
9Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/v
Ä
MAdam/res_block_1/cnn_block_7/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp9Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/v*
_output_shapes	
:*
dtype0
Í
:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/v
Æ
NAdam/res_block_1/cnn_block_7/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/v*
_output_shapes	
:*
dtype0
±
,Adam/res_block_1/cnn_block_7/conv2d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/res_block_1/cnn_block_7/conv2d_8/bias/v
ª
@Adam/res_block_1/cnn_block_7/conv2d_8/bias/v/Read/ReadVariableOpReadVariableOp,Adam/res_block_1/cnn_block_7/conv2d_8/bias/v*
_output_shapes	
:*
dtype0
Â
.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/v
»
BAdam/res_block_1/cnn_block_7/conv2d_8/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/v*(
_output_shapes
:*
dtype0
Ë
9Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/v
Ä
MAdam/res_block_1/cnn_block_6/batch_normalization_6/beta/v/Read/ReadVariableOpReadVariableOp9Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/v*
_output_shapes	
:*
dtype0
Í
:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/v
Æ
NAdam/res_block_1/cnn_block_6/batch_normalization_6/gamma/v/Read/ReadVariableOpReadVariableOp:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/v*
_output_shapes	
:*
dtype0
±
,Adam/res_block_1/cnn_block_6/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/res_block_1/cnn_block_6/conv2d_7/bias/v
ª
@Adam/res_block_1/cnn_block_6/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOp,Adam/res_block_1/cnn_block_6/conv2d_7/bias/v*
_output_shapes	
:*
dtype0
Á
.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/v
º
BAdam/res_block_1/cnn_block_6/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/v*'
_output_shapes
:@*
dtype0

Adam/res_block/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/res_block/conv2d_6/bias/v

2Adam/res_block/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/res_block/conv2d_6/bias/v*
_output_shapes
: *
dtype0
¤
 Adam/res_block/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/res_block/conv2d_6/kernel/v

4Adam/res_block/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOp Adam/res_block/conv2d_6/kernel/v*&
_output_shapes
: *
dtype0
Æ
7Adam/res_block/cnn_block_5/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97Adam/res_block/cnn_block_5/batch_normalization_5/beta/v
¿
KAdam/res_block/cnn_block_5/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp7Adam/res_block/cnn_block_5/batch_normalization_5/beta/v*
_output_shapes
:@*
dtype0
È
8Adam/res_block/cnn_block_5/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8Adam/res_block/cnn_block_5/batch_normalization_5/gamma/v
Á
LAdam/res_block/cnn_block_5/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp8Adam/res_block/cnn_block_5/batch_normalization_5/gamma/v*
_output_shapes
:@*
dtype0
¬
*Adam/res_block/cnn_block_5/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/res_block/cnn_block_5/conv2d_5/bias/v
¥
>Adam/res_block/cnn_block_5/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOp*Adam/res_block/cnn_block_5/conv2d_5/bias/v*
_output_shapes
:@*
dtype0
¼
,Adam/res_block/cnn_block_5/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*=
shared_name.,Adam/res_block/cnn_block_5/conv2d_5/kernel/v
µ
@Adam/res_block/cnn_block_5/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/res_block/cnn_block_5/conv2d_5/kernel/v*&
_output_shapes
: @*
dtype0
Æ
7Adam/res_block/cnn_block_4/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/res_block/cnn_block_4/batch_normalization_4/beta/v
¿
KAdam/res_block/cnn_block_4/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp7Adam/res_block/cnn_block_4/batch_normalization_4/beta/v*
_output_shapes
: *
dtype0
È
8Adam/res_block/cnn_block_4/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adam/res_block/cnn_block_4/batch_normalization_4/gamma/v
Á
LAdam/res_block/cnn_block_4/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp8Adam/res_block/cnn_block_4/batch_normalization_4/gamma/v*
_output_shapes
: *
dtype0
¬
*Adam/res_block/cnn_block_4/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/res_block/cnn_block_4/conv2d_4/bias/v
¥
>Adam/res_block/cnn_block_4/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOp*Adam/res_block/cnn_block_4/conv2d_4/bias/v*
_output_shapes
: *
dtype0
¼
,Adam/res_block/cnn_block_4/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *=
shared_name.,Adam/res_block/cnn_block_4/conv2d_4/kernel/v
µ
@Adam/res_block/cnn_block_4/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/res_block/cnn_block_4/conv2d_4/kernel/v*&
_output_shapes
:  *
dtype0
Æ
7Adam/res_block/cnn_block_3/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/res_block/cnn_block_3/batch_normalization_3/beta/v
¿
KAdam/res_block/cnn_block_3/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp7Adam/res_block/cnn_block_3/batch_normalization_3/beta/v*
_output_shapes
: *
dtype0
È
8Adam/res_block/cnn_block_3/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adam/res_block/cnn_block_3/batch_normalization_3/gamma/v
Á
LAdam/res_block/cnn_block_3/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp8Adam/res_block/cnn_block_3/batch_normalization_3/gamma/v*
_output_shapes
: *
dtype0
¬
*Adam/res_block/cnn_block_3/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/res_block/cnn_block_3/conv2d_3/bias/v
¥
>Adam/res_block/cnn_block_3/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOp*Adam/res_block/cnn_block_3/conv2d_3/bias/v*
_output_shapes
: *
dtype0
¼
,Adam/res_block/cnn_block_3/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,Adam/res_block/cnn_block_3/conv2d_3/kernel/v
µ
@Adam/res_block/cnn_block_3/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/res_block/cnn_block_3/conv2d_3/kernel/v*&
_output_shapes
: *
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:
*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	b
*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes
:	b
*
dtype0

!Adam/res_block_1/conv2d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/res_block_1/conv2d_10/bias/m

5Adam/res_block_1/conv2d_10/bias/m/Read/ReadVariableOpReadVariableOp!Adam/res_block_1/conv2d_10/bias/m*
_output_shapes	
:*
dtype0
«
#Adam/res_block_1/conv2d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/res_block_1/conv2d_10/kernel/m
¤
7Adam/res_block_1/conv2d_10/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/res_block_1/conv2d_10/kernel/m*'
_output_shapes
:@*
dtype0
Ë
9Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/m
Ä
MAdam/res_block_1/cnn_block_8/batch_normalization_8/beta/m/Read/ReadVariableOpReadVariableOp9Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/m*
_output_shapes	
:*
dtype0
Í
:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/m
Æ
NAdam/res_block_1/cnn_block_8/batch_normalization_8/gamma/m/Read/ReadVariableOpReadVariableOp:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/m*
_output_shapes	
:*
dtype0
±
,Adam/res_block_1/cnn_block_8/conv2d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/res_block_1/cnn_block_8/conv2d_9/bias/m
ª
@Adam/res_block_1/cnn_block_8/conv2d_9/bias/m/Read/ReadVariableOpReadVariableOp,Adam/res_block_1/cnn_block_8/conv2d_9/bias/m*
_output_shapes	
:*
dtype0
Â
.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/m
»
BAdam/res_block_1/cnn_block_8/conv2d_9/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/m*(
_output_shapes
:*
dtype0
Ë
9Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/m
Ä
MAdam/res_block_1/cnn_block_7/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp9Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/m*
_output_shapes	
:*
dtype0
Í
:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/m
Æ
NAdam/res_block_1/cnn_block_7/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/m*
_output_shapes	
:*
dtype0
±
,Adam/res_block_1/cnn_block_7/conv2d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/res_block_1/cnn_block_7/conv2d_8/bias/m
ª
@Adam/res_block_1/cnn_block_7/conv2d_8/bias/m/Read/ReadVariableOpReadVariableOp,Adam/res_block_1/cnn_block_7/conv2d_8/bias/m*
_output_shapes	
:*
dtype0
Â
.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/m
»
BAdam/res_block_1/cnn_block_7/conv2d_8/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/m*(
_output_shapes
:*
dtype0
Ë
9Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/m
Ä
MAdam/res_block_1/cnn_block_6/batch_normalization_6/beta/m/Read/ReadVariableOpReadVariableOp9Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/m*
_output_shapes	
:*
dtype0
Í
:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/m
Æ
NAdam/res_block_1/cnn_block_6/batch_normalization_6/gamma/m/Read/ReadVariableOpReadVariableOp:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/m*
_output_shapes	
:*
dtype0
±
,Adam/res_block_1/cnn_block_6/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/res_block_1/cnn_block_6/conv2d_7/bias/m
ª
@Adam/res_block_1/cnn_block_6/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOp,Adam/res_block_1/cnn_block_6/conv2d_7/bias/m*
_output_shapes	
:*
dtype0
Á
.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/m
º
BAdam/res_block_1/cnn_block_6/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/m*'
_output_shapes
:@*
dtype0

Adam/res_block/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/res_block/conv2d_6/bias/m

2Adam/res_block/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/res_block/conv2d_6/bias/m*
_output_shapes
: *
dtype0
¤
 Adam/res_block/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/res_block/conv2d_6/kernel/m

4Adam/res_block/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOp Adam/res_block/conv2d_6/kernel/m*&
_output_shapes
: *
dtype0
Æ
7Adam/res_block/cnn_block_5/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97Adam/res_block/cnn_block_5/batch_normalization_5/beta/m
¿
KAdam/res_block/cnn_block_5/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp7Adam/res_block/cnn_block_5/batch_normalization_5/beta/m*
_output_shapes
:@*
dtype0
È
8Adam/res_block/cnn_block_5/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8Adam/res_block/cnn_block_5/batch_normalization_5/gamma/m
Á
LAdam/res_block/cnn_block_5/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp8Adam/res_block/cnn_block_5/batch_normalization_5/gamma/m*
_output_shapes
:@*
dtype0
¬
*Adam/res_block/cnn_block_5/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/res_block/cnn_block_5/conv2d_5/bias/m
¥
>Adam/res_block/cnn_block_5/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOp*Adam/res_block/cnn_block_5/conv2d_5/bias/m*
_output_shapes
:@*
dtype0
¼
,Adam/res_block/cnn_block_5/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*=
shared_name.,Adam/res_block/cnn_block_5/conv2d_5/kernel/m
µ
@Adam/res_block/cnn_block_5/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/res_block/cnn_block_5/conv2d_5/kernel/m*&
_output_shapes
: @*
dtype0
Æ
7Adam/res_block/cnn_block_4/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/res_block/cnn_block_4/batch_normalization_4/beta/m
¿
KAdam/res_block/cnn_block_4/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp7Adam/res_block/cnn_block_4/batch_normalization_4/beta/m*
_output_shapes
: *
dtype0
È
8Adam/res_block/cnn_block_4/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adam/res_block/cnn_block_4/batch_normalization_4/gamma/m
Á
LAdam/res_block/cnn_block_4/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp8Adam/res_block/cnn_block_4/batch_normalization_4/gamma/m*
_output_shapes
: *
dtype0
¬
*Adam/res_block/cnn_block_4/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/res_block/cnn_block_4/conv2d_4/bias/m
¥
>Adam/res_block/cnn_block_4/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOp*Adam/res_block/cnn_block_4/conv2d_4/bias/m*
_output_shapes
: *
dtype0
¼
,Adam/res_block/cnn_block_4/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *=
shared_name.,Adam/res_block/cnn_block_4/conv2d_4/kernel/m
µ
@Adam/res_block/cnn_block_4/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/res_block/cnn_block_4/conv2d_4/kernel/m*&
_output_shapes
:  *
dtype0
Æ
7Adam/res_block/cnn_block_3/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/res_block/cnn_block_3/batch_normalization_3/beta/m
¿
KAdam/res_block/cnn_block_3/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp7Adam/res_block/cnn_block_3/batch_normalization_3/beta/m*
_output_shapes
: *
dtype0
È
8Adam/res_block/cnn_block_3/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adam/res_block/cnn_block_3/batch_normalization_3/gamma/m
Á
LAdam/res_block/cnn_block_3/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp8Adam/res_block/cnn_block_3/batch_normalization_3/gamma/m*
_output_shapes
: *
dtype0
¬
*Adam/res_block/cnn_block_3/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/res_block/cnn_block_3/conv2d_3/bias/m
¥
>Adam/res_block/cnn_block_3/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOp*Adam/res_block/cnn_block_3/conv2d_3/bias/m*
_output_shapes
: *
dtype0
¼
,Adam/res_block/cnn_block_3/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,Adam/res_block/cnn_block_3/conv2d_3/kernel/m
µ
@Adam/res_block/cnn_block_3/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/res_block/cnn_block_3/conv2d_3/kernel/m*&
_output_shapes
: *
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:
*
dtype0

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	b
*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes
:	b
*
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
Ó
=res_block_1/cnn_block_8/batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=res_block_1/cnn_block_8/batch_normalization_8/moving_variance
Ì
Qres_block_1/cnn_block_8/batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp=res_block_1/cnn_block_8/batch_normalization_8/moving_variance*
_output_shapes	
:*
dtype0
Ë
9res_block_1/cnn_block_8/batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9res_block_1/cnn_block_8/batch_normalization_8/moving_mean
Ä
Mres_block_1/cnn_block_8/batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp9res_block_1/cnn_block_8/batch_normalization_8/moving_mean*
_output_shapes	
:*
dtype0
Ó
=res_block_1/cnn_block_7/batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=res_block_1/cnn_block_7/batch_normalization_7/moving_variance
Ì
Qres_block_1/cnn_block_7/batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp=res_block_1/cnn_block_7/batch_normalization_7/moving_variance*
_output_shapes	
:*
dtype0
Ë
9res_block_1/cnn_block_7/batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9res_block_1/cnn_block_7/batch_normalization_7/moving_mean
Ä
Mres_block_1/cnn_block_7/batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp9res_block_1/cnn_block_7/batch_normalization_7/moving_mean*
_output_shapes	
:*
dtype0
Ó
=res_block_1/cnn_block_6/batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=res_block_1/cnn_block_6/batch_normalization_6/moving_variance
Ì
Qres_block_1/cnn_block_6/batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp=res_block_1/cnn_block_6/batch_normalization_6/moving_variance*
_output_shapes	
:*
dtype0
Ë
9res_block_1/cnn_block_6/batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9res_block_1/cnn_block_6/batch_normalization_6/moving_mean
Ä
Mres_block_1/cnn_block_6/batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp9res_block_1/cnn_block_6/batch_normalization_6/moving_mean*
_output_shapes	
:*
dtype0

res_block_1/conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameres_block_1/conv2d_10/bias

.res_block_1/conv2d_10/bias/Read/ReadVariableOpReadVariableOpres_block_1/conv2d_10/bias*
_output_shapes	
:*
dtype0

res_block_1/conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameres_block_1/conv2d_10/kernel

0res_block_1/conv2d_10/kernel/Read/ReadVariableOpReadVariableOpres_block_1/conv2d_10/kernel*'
_output_shapes
:@*
dtype0
½
2res_block_1/cnn_block_8/batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42res_block_1/cnn_block_8/batch_normalization_8/beta
¶
Fres_block_1/cnn_block_8/batch_normalization_8/beta/Read/ReadVariableOpReadVariableOp2res_block_1/cnn_block_8/batch_normalization_8/beta*
_output_shapes	
:*
dtype0
¿
3res_block_1/cnn_block_8/batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53res_block_1/cnn_block_8/batch_normalization_8/gamma
¸
Gres_block_1/cnn_block_8/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOp3res_block_1/cnn_block_8/batch_normalization_8/gamma*
_output_shapes	
:*
dtype0
£
%res_block_1/cnn_block_8/conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%res_block_1/cnn_block_8/conv2d_9/bias

9res_block_1/cnn_block_8/conv2d_9/bias/Read/ReadVariableOpReadVariableOp%res_block_1/cnn_block_8/conv2d_9/bias*
_output_shapes	
:*
dtype0
´
'res_block_1/cnn_block_8/conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'res_block_1/cnn_block_8/conv2d_9/kernel
­
;res_block_1/cnn_block_8/conv2d_9/kernel/Read/ReadVariableOpReadVariableOp'res_block_1/cnn_block_8/conv2d_9/kernel*(
_output_shapes
:*
dtype0
½
2res_block_1/cnn_block_7/batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42res_block_1/cnn_block_7/batch_normalization_7/beta
¶
Fres_block_1/cnn_block_7/batch_normalization_7/beta/Read/ReadVariableOpReadVariableOp2res_block_1/cnn_block_7/batch_normalization_7/beta*
_output_shapes	
:*
dtype0
¿
3res_block_1/cnn_block_7/batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53res_block_1/cnn_block_7/batch_normalization_7/gamma
¸
Gres_block_1/cnn_block_7/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOp3res_block_1/cnn_block_7/batch_normalization_7/gamma*
_output_shapes	
:*
dtype0
£
%res_block_1/cnn_block_7/conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%res_block_1/cnn_block_7/conv2d_8/bias

9res_block_1/cnn_block_7/conv2d_8/bias/Read/ReadVariableOpReadVariableOp%res_block_1/cnn_block_7/conv2d_8/bias*
_output_shapes	
:*
dtype0
´
'res_block_1/cnn_block_7/conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'res_block_1/cnn_block_7/conv2d_8/kernel
­
;res_block_1/cnn_block_7/conv2d_8/kernel/Read/ReadVariableOpReadVariableOp'res_block_1/cnn_block_7/conv2d_8/kernel*(
_output_shapes
:*
dtype0
½
2res_block_1/cnn_block_6/batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42res_block_1/cnn_block_6/batch_normalization_6/beta
¶
Fres_block_1/cnn_block_6/batch_normalization_6/beta/Read/ReadVariableOpReadVariableOp2res_block_1/cnn_block_6/batch_normalization_6/beta*
_output_shapes	
:*
dtype0
¿
3res_block_1/cnn_block_6/batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53res_block_1/cnn_block_6/batch_normalization_6/gamma
¸
Gres_block_1/cnn_block_6/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOp3res_block_1/cnn_block_6/batch_normalization_6/gamma*
_output_shapes	
:*
dtype0
£
%res_block_1/cnn_block_6/conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%res_block_1/cnn_block_6/conv2d_7/bias

9res_block_1/cnn_block_6/conv2d_7/bias/Read/ReadVariableOpReadVariableOp%res_block_1/cnn_block_6/conv2d_7/bias*
_output_shapes	
:*
dtype0
³
'res_block_1/cnn_block_6/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'res_block_1/cnn_block_6/conv2d_7/kernel
¬
;res_block_1/cnn_block_6/conv2d_7/kernel/Read/ReadVariableOpReadVariableOp'res_block_1/cnn_block_6/conv2d_7/kernel*'
_output_shapes
:@*
dtype0
Î
;res_block/cnn_block_5/batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*L
shared_name=;res_block/cnn_block_5/batch_normalization_5/moving_variance
Ç
Ores_block/cnn_block_5/batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp;res_block/cnn_block_5/batch_normalization_5/moving_variance*
_output_shapes
:@*
dtype0
Æ
7res_block/cnn_block_5/batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97res_block/cnn_block_5/batch_normalization_5/moving_mean
¿
Kres_block/cnn_block_5/batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp7res_block/cnn_block_5/batch_normalization_5/moving_mean*
_output_shapes
:@*
dtype0
Î
;res_block/cnn_block_4/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;res_block/cnn_block_4/batch_normalization_4/moving_variance
Ç
Ores_block/cnn_block_4/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp;res_block/cnn_block_4/batch_normalization_4/moving_variance*
_output_shapes
: *
dtype0
Æ
7res_block/cnn_block_4/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97res_block/cnn_block_4/batch_normalization_4/moving_mean
¿
Kres_block/cnn_block_4/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp7res_block/cnn_block_4/batch_normalization_4/moving_mean*
_output_shapes
: *
dtype0
Î
;res_block/cnn_block_3/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;res_block/cnn_block_3/batch_normalization_3/moving_variance
Ç
Ores_block/cnn_block_3/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp;res_block/cnn_block_3/batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0
Æ
7res_block/cnn_block_3/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97res_block/cnn_block_3/batch_normalization_3/moving_mean
¿
Kres_block/cnn_block_3/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp7res_block/cnn_block_3/batch_normalization_3/moving_mean*
_output_shapes
: *
dtype0

res_block/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameres_block/conv2d_6/bias

+res_block/conv2d_6/bias/Read/ReadVariableOpReadVariableOpres_block/conv2d_6/bias*
_output_shapes
: *
dtype0

res_block/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameres_block/conv2d_6/kernel

-res_block/conv2d_6/kernel/Read/ReadVariableOpReadVariableOpres_block/conv2d_6/kernel*&
_output_shapes
: *
dtype0
¸
0res_block/cnn_block_5/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20res_block/cnn_block_5/batch_normalization_5/beta
±
Dres_block/cnn_block_5/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp0res_block/cnn_block_5/batch_normalization_5/beta*
_output_shapes
:@*
dtype0
º
1res_block/cnn_block_5/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31res_block/cnn_block_5/batch_normalization_5/gamma
³
Eres_block/cnn_block_5/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp1res_block/cnn_block_5/batch_normalization_5/gamma*
_output_shapes
:@*
dtype0

#res_block/cnn_block_5/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#res_block/cnn_block_5/conv2d_5/bias

7res_block/cnn_block_5/conv2d_5/bias/Read/ReadVariableOpReadVariableOp#res_block/cnn_block_5/conv2d_5/bias*
_output_shapes
:@*
dtype0
®
%res_block/cnn_block_5/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*6
shared_name'%res_block/cnn_block_5/conv2d_5/kernel
§
9res_block/cnn_block_5/conv2d_5/kernel/Read/ReadVariableOpReadVariableOp%res_block/cnn_block_5/conv2d_5/kernel*&
_output_shapes
: @*
dtype0
¸
0res_block/cnn_block_4/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20res_block/cnn_block_4/batch_normalization_4/beta
±
Dres_block/cnn_block_4/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp0res_block/cnn_block_4/batch_normalization_4/beta*
_output_shapes
: *
dtype0
º
1res_block/cnn_block_4/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31res_block/cnn_block_4/batch_normalization_4/gamma
³
Eres_block/cnn_block_4/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp1res_block/cnn_block_4/batch_normalization_4/gamma*
_output_shapes
: *
dtype0

#res_block/cnn_block_4/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#res_block/cnn_block_4/conv2d_4/bias

7res_block/cnn_block_4/conv2d_4/bias/Read/ReadVariableOpReadVariableOp#res_block/cnn_block_4/conv2d_4/bias*
_output_shapes
: *
dtype0
®
%res_block/cnn_block_4/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *6
shared_name'%res_block/cnn_block_4/conv2d_4/kernel
§
9res_block/cnn_block_4/conv2d_4/kernel/Read/ReadVariableOpReadVariableOp%res_block/cnn_block_4/conv2d_4/kernel*&
_output_shapes
:  *
dtype0
¸
0res_block/cnn_block_3/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20res_block/cnn_block_3/batch_normalization_3/beta
±
Dres_block/cnn_block_3/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp0res_block/cnn_block_3/batch_normalization_3/beta*
_output_shapes
: *
dtype0
º
1res_block/cnn_block_3/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31res_block/cnn_block_3/batch_normalization_3/gamma
³
Eres_block/cnn_block_3/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp1res_block/cnn_block_3/batch_normalization_3/gamma*
_output_shapes
: *
dtype0

#res_block/cnn_block_3/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#res_block/cnn_block_3/conv2d_3/bias

7res_block/cnn_block_3/conv2d_3/bias/Read/ReadVariableOpReadVariableOp#res_block/cnn_block_3/conv2d_3/bias*
_output_shapes
: *
dtype0
®
%res_block/cnn_block_3/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%res_block/cnn_block_3/conv2d_3/kernel
§
9res_block/cnn_block_3/conv2d_3/kernel/Read/ReadVariableOpReadVariableOp%res_block/cnn_block_3/conv2d_3/kernel*&
_output_shapes
: *
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:
*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	b
*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	b
*
dtype0

NoOpNoOp
ñ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*«
value B B
Û
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
ß
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
channels
cnn1
cnn2
cnn3
pooling
identity_mapping*
ß
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
!channels
"cnn1
#cnn2
$cnn3
%pooling
&identity_mapping*

'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses* 
¦
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias*
Ê
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
D15
E16
F17
G18
H19
I20
J21
K22
L23
M24
N25
O26
P27
Q28
R29
S30
T31
U32
V33
W34
X35
Y36
Z37
[38
\39
340
441*
ê
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
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
U26
V27
328
429*
* 
°
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
btrace_0
ctrace_1
dtrace_2
etrace_3* 
6
ftrace_0
gtrace_1
htrace_2
itrace_3* 
* 

jiter

kbeta_1

lbeta_2
	mdecay
nlearning_rate3mÃ4mÄ5mÅ6mÆ7mÇ8mÈ9mÉ:mÊ;mË<mÌ=mÍ>mÎ?mÏ@mÐAmÑBmÒImÓJmÔKmÕLmÖMm×NmØOmÙPmÚQmÛRmÜSmÝTmÞUmßVmà3vá4vâ5vã6vä7vå8væ9vç:vè;vé<vê=vë>vì?ví@vîAvïBvðIvñJvòKvóLvôMvõNvöOv÷PvøQvùRvúSvûTvüUvýVvþ*

oserving_default* 

50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
D15
E16
F17
G18
H19*
j
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13*
* 

pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

utrace_0
vtrace_1* 

wtrace_0
xtrace_1* 
* 
£
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses
conv
bn*
ª
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	conv
bn*
ª
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	conv
bn*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
Ï
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

Akernel
Bbias
!_jit_compiled_convolution_op*

I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13
W14
X15
Y16
Z17
[18
\19*
j
I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13*
* 

non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*

£trace_0
¤trace_1* 

¥trace_0
¦trace_1* 
* 
ª
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«__call__
+¬&call_and_return_all_conditional_losses
	­conv
®bn*
ª
¯	variables
°trainable_variables
±regularization_losses
²	keras_api
³__call__
+´&call_and_return_all_conditional_losses
	µconv
¶bn*
ª
·	variables
¸trainable_variables
¹regularization_losses
º	keras_api
»__call__
+¼&call_and_return_all_conditional_losses
	½conv
¾bn*

¿	variables
Àtrainable_variables
Áregularization_losses
Â	keras_api
Ã__call__
+Ä&call_and_return_all_conditional_losses* 
Ï
Å	variables
Ætrainable_variables
Çregularization_losses
È	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses

Ukernel
Vbias
!Ë_jit_compiled_convolution_op*
* 
* 
* 

Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses* 

Ñtrace_0* 

Òtrace_0* 

30
41*

30
41*
* 

Ónon_trainable_variables
Ôlayers
Õmetrics
 Ölayer_regularization_losses
×layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

Øtrace_0* 

Ùtrace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%res_block/cnn_block_3/conv2d_3/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#res_block/cnn_block_3/conv2d_3/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1res_block/cnn_block_3/batch_normalization_3/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE0res_block/cnn_block_3/batch_normalization_3/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%res_block/cnn_block_4/conv2d_4/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#res_block/cnn_block_4/conv2d_4/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1res_block/cnn_block_4/batch_normalization_4/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE0res_block/cnn_block_4/batch_normalization_4/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%res_block/cnn_block_5/conv2d_5/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#res_block/cnn_block_5/conv2d_5/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1res_block/cnn_block_5/batch_normalization_5/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE0res_block/cnn_block_5/batch_normalization_5/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEres_block/conv2d_6/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEres_block/conv2d_6/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE7res_block/cnn_block_3/batch_normalization_3/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;res_block/cnn_block_3/batch_normalization_3/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE7res_block/cnn_block_4/batch_normalization_4/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;res_block/cnn_block_4/batch_normalization_4/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE7res_block/cnn_block_5/batch_normalization_5/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;res_block/cnn_block_5/batch_normalization_5/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'res_block_1/cnn_block_6/conv2d_7/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%res_block_1/cnn_block_6/conv2d_7/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3res_block_1/cnn_block_6/batch_normalization_6/gamma'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2res_block_1/cnn_block_6/batch_normalization_6/beta'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'res_block_1/cnn_block_7/conv2d_8/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%res_block_1/cnn_block_7/conv2d_8/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3res_block_1/cnn_block_7/batch_normalization_7/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2res_block_1/cnn_block_7/batch_normalization_7/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'res_block_1/cnn_block_8/conv2d_9/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%res_block_1/cnn_block_8/conv2d_9/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3res_block_1/cnn_block_8/batch_normalization_8/gamma'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2res_block_1/cnn_block_8/batch_normalization_8/beta'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEres_block_1/conv2d_10/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEres_block_1/conv2d_10/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE9res_block_1/cnn_block_6/batch_normalization_6/moving_mean'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=res_block_1/cnn_block_6/batch_normalization_6/moving_variance'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE9res_block_1/cnn_block_7/batch_normalization_7/moving_mean'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=res_block_1/cnn_block_7/batch_normalization_7/moving_variance'variables/37/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE9res_block_1/cnn_block_8/batch_normalization_8/moving_mean'variables/38/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=res_block_1/cnn_block_8/batch_normalization_8/moving_variance'variables/39/.ATTRIBUTES/VARIABLE_VALUE*
Z
C0
D1
E2
F3
G4
H5
W6
X7
Y8
Z9
[10
\11*
'
0
1
2
3
4*

Ú0
Û1*
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
.
C0
D1
E2
F3
G4
H5*
'
0
1
2
3
4*
* 
* 
* 
* 
* 
* 
* 
.
50
61
72
83
C4
D5*
 
50
61
72
83*
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
Ï
á	variables
âtrainable_variables
ãregularization_losses
ä	keras_api
å__call__
+æ&call_and_return_all_conditional_losses

5kernel
6bias
!ç_jit_compiled_convolution_op*
Ü
è	variables
étrainable_variables
êregularization_losses
ë	keras_api
ì__call__
+í&call_and_return_all_conditional_losses
	îaxis
	7gamma
8beta
Cmoving_mean
Dmoving_variance*
.
90
:1
;2
<3
E4
F5*
 
90
:1
;2
<3*
* 

ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
Ï
ô	variables
õtrainable_variables
öregularization_losses
÷	keras_api
ø__call__
+ù&call_and_return_all_conditional_losses

9kernel
:bias
!ú_jit_compiled_convolution_op*
Ü
û	variables
ütrainable_variables
ýregularization_losses
þ	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses
	axis
	;gamma
<beta
Emoving_mean
Fmoving_variance*
.
=0
>1
?2
@3
G4
H5*
 
=0
>1
?2
@3*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
Ï
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

=kernel
>bias
!_jit_compiled_convolution_op*
Ü
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis
	?gamma
@beta
Gmoving_mean
Hmoving_variance*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

A0
B1*

A0
B1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
.
W0
X1
Y2
Z3
[4
\5*
'
"0
#1
$2
%3
&4*
* 
* 
* 
* 
* 
* 
* 
.
I0
J1
K2
L3
W4
X5*
 
I0
J1
K2
L3*
* 

¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses*
* 
* 
Ï
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
ª__call__
+«&call_and_return_all_conditional_losses

Ikernel
Jbias
!¬_jit_compiled_convolution_op*
Ü
­	variables
®trainable_variables
¯regularization_losses
°	keras_api
±__call__
+²&call_and_return_all_conditional_losses
	³axis
	Kgamma
Lbeta
Wmoving_mean
Xmoving_variance*
.
M0
N1
O2
P3
Y4
Z5*
 
M0
N1
O2
P3*
* 

´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
¯	variables
°trainable_variables
±regularization_losses
³__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses*
* 
* 
Ï
¹	variables
ºtrainable_variables
»regularization_losses
¼	keras_api
½__call__
+¾&call_and_return_all_conditional_losses

Mkernel
Nbias
!¿_jit_compiled_convolution_op*
Ü
À	variables
Átrainable_variables
Âregularization_losses
Ã	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses
	Æaxis
	Ogamma
Pbeta
Ymoving_mean
Zmoving_variance*
.
Q0
R1
S2
T3
[4
\5*
 
Q0
R1
S2
T3*
* 

Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
·	variables
¸trainable_variables
¹regularization_losses
»__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses*
* 
* 
Ï
Ì	variables
Ítrainable_variables
Îregularization_losses
Ï	keras_api
Ð__call__
+Ñ&call_and_return_all_conditional_losses

Qkernel
Rbias
!Ò_jit_compiled_convolution_op*
Ü
Ó	variables
Ôtrainable_variables
Õregularization_losses
Ö	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses
	Ùaxis
	Sgamma
Tbeta
[moving_mean
\moving_variance*
* 
* 
* 

Únon_trainable_variables
Ûlayers
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
¿	variables
Àtrainable_variables
Áregularization_losses
Ã__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses* 

ßtrace_0* 

àtrace_0* 

U0
V1*

U0
V1*
* 

ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
Å	variables
Ætrainable_variables
Çregularization_losses
É__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses*
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
<
æ	variables
ç	keras_api

ètotal

écount*
M
ê	variables
ë	keras_api

ìtotal

ícount
î
_fn_kwargs*

C0
D1*

0
1*
* 
* 
* 

50
61*

50
61*
* 

ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
á	variables
âtrainable_variables
ãregularization_losses
å__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses*
* 
* 
* 
 
70
81
C2
D3*

70
81*
* 

ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
è	variables
étrainable_variables
êregularization_losses
ì__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses*

ùtrace_0
útrace_1* 

ûtrace_0
ütrace_1* 
* 

E0
F1*

0
1*
* 
* 
* 

90
:1*

90
:1*
* 

ýnon_trainable_variables
þlayers
ÿmetrics
 layer_regularization_losses
layer_metrics
ô	variables
õtrainable_variables
öregularization_losses
ø__call__
+ù&call_and_return_all_conditional_losses
'ù"call_and_return_conditional_losses*
* 
* 
* 
 
;0
<1
E2
F3*

;0
<1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
û	variables
ütrainable_variables
ýregularization_losses
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 

G0
H1*

0
1*
* 
* 
* 

=0
>1*

=0
>1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
 
?0
@1
G2
H3*

?0
@1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
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
W0
X1*

­0
®1*
* 
* 
* 

I0
J1*

I0
J1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¦	variables
§trainable_variables
¨regularization_losses
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses*
* 
* 
* 
 
K0
L1
W2
X3*

K0
L1*
* 

non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
­	variables
®trainable_variables
¯regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses*

£trace_0
¤trace_1* 

¥trace_0
¦trace_1* 
* 

Y0
Z1*

µ0
¶1*
* 
* 
* 

M0
N1*

M0
N1*
* 

§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
¹	variables
ºtrainable_variables
»regularization_losses
½__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses*
* 
* 
* 
 
O0
P1
Y2
Z3*

O0
P1*
* 

¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
À	variables
Átrainable_variables
Âregularization_losses
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses*

±trace_0
²trace_1* 

³trace_0
´trace_1* 
* 

[0
\1*

½0
¾1*
* 
* 
* 

Q0
R1*

Q0
R1*
* 

µnon_trainable_variables
¶layers
·metrics
 ¸layer_regularization_losses
¹layer_metrics
Ì	variables
Ítrainable_variables
Îregularization_losses
Ð__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses*
* 
* 
* 
 
S0
T1
[2
\3*

S0
T1*
* 

ºnon_trainable_variables
»layers
¼metrics
 ½layer_regularization_losses
¾layer_metrics
Ó	variables
Ôtrainable_variables
Õregularization_losses
×__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses*

¿trace_0
Àtrace_1* 

Átrace_0
Âtrace_1* 
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

è0
é1*

æ	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

ì0
í1*

ê	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 

C0
D1*
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
E0
F1*
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
G0
H1*
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
W0
X1*
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
Y0
Z1*
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
[0
\1*
* 
* 
* 
* 
* 
* 
* 
* 
{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/res_block/cnn_block_3/conv2d_3/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/res_block/cnn_block_3/conv2d_3/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/res_block/cnn_block_3/batch_normalization_3/gamma/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE7Adam/res_block/cnn_block_3/batch_normalization_3/beta/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/res_block/cnn_block_4/conv2d_4/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/res_block/cnn_block_4/conv2d_4/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/res_block/cnn_block_4/batch_normalization_4/gamma/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE7Adam/res_block/cnn_block_4/batch_normalization_4/beta/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/res_block/cnn_block_5/conv2d_5/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/res_block/cnn_block_5/conv2d_5/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/res_block/cnn_block_5/batch_normalization_5/gamma/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE7Adam/res_block/cnn_block_5/batch_normalization_5/beta/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/res_block/conv2d_6/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/res_block/conv2d_6/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/res_block_1/cnn_block_6/conv2d_7/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE9Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/res_block_1/cnn_block_7/conv2d_8/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE9Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/mCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/res_block_1/cnn_block_8/conv2d_9/bias/mCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/mCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE9Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/mCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE#Adam/res_block_1/conv2d_10/kernel/mCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/res_block_1/conv2d_10/bias/mCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/res_block/cnn_block_3/conv2d_3/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/res_block/cnn_block_3/conv2d_3/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/res_block/cnn_block_3/batch_normalization_3/gamma/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE7Adam/res_block/cnn_block_3/batch_normalization_3/beta/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/res_block/cnn_block_4/conv2d_4/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/res_block/cnn_block_4/conv2d_4/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/res_block/cnn_block_4/batch_normalization_4/gamma/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE7Adam/res_block/cnn_block_4/batch_normalization_4/beta/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/res_block/cnn_block_5/conv2d_5/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/res_block/cnn_block_5/conv2d_5/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/res_block/cnn_block_5/batch_normalization_5/gamma/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE7Adam/res_block/cnn_block_5/batch_normalization_5/beta/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/res_block/conv2d_6/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/res_block/conv2d_6/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/res_block_1/cnn_block_6/conv2d_7/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE9Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/res_block_1/cnn_block_7/conv2d_8/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE9Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/vCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/res_block_1/cnn_block_8/conv2d_9/bias/vCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/vCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE9Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/vCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE#Adam/res_block_1/conv2d_10/kernel/vCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/res_block_1/conv2d_10/bias/vCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_1Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
£
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1%res_block/cnn_block_3/conv2d_3/kernel#res_block/cnn_block_3/conv2d_3/bias1res_block/cnn_block_3/batch_normalization_3/gamma0res_block/cnn_block_3/batch_normalization_3/beta7res_block/cnn_block_3/batch_normalization_3/moving_mean;res_block/cnn_block_3/batch_normalization_3/moving_variance%res_block/cnn_block_4/conv2d_4/kernel#res_block/cnn_block_4/conv2d_4/bias1res_block/cnn_block_4/batch_normalization_4/gamma0res_block/cnn_block_4/batch_normalization_4/beta7res_block/cnn_block_4/batch_normalization_4/moving_mean;res_block/cnn_block_4/batch_normalization_4/moving_varianceres_block/conv2d_6/kernelres_block/conv2d_6/bias%res_block/cnn_block_5/conv2d_5/kernel#res_block/cnn_block_5/conv2d_5/bias1res_block/cnn_block_5/batch_normalization_5/gamma0res_block/cnn_block_5/batch_normalization_5/beta7res_block/cnn_block_5/batch_normalization_5/moving_mean;res_block/cnn_block_5/batch_normalization_5/moving_variance'res_block_1/cnn_block_6/conv2d_7/kernel%res_block_1/cnn_block_6/conv2d_7/bias3res_block_1/cnn_block_6/batch_normalization_6/gamma2res_block_1/cnn_block_6/batch_normalization_6/beta9res_block_1/cnn_block_6/batch_normalization_6/moving_mean=res_block_1/cnn_block_6/batch_normalization_6/moving_variance'res_block_1/cnn_block_7/conv2d_8/kernel%res_block_1/cnn_block_7/conv2d_8/bias3res_block_1/cnn_block_7/batch_normalization_7/gamma2res_block_1/cnn_block_7/batch_normalization_7/beta9res_block_1/cnn_block_7/batch_normalization_7/moving_mean=res_block_1/cnn_block_7/batch_normalization_7/moving_varianceres_block_1/conv2d_10/kernelres_block_1/conv2d_10/bias'res_block_1/cnn_block_8/conv2d_9/kernel%res_block_1/cnn_block_8/conv2d_9/bias3res_block_1/cnn_block_8/batch_normalization_8/gamma2res_block_1/cnn_block_8/batch_normalization_8/beta9res_block_1/cnn_block_8/batch_normalization_8/moving_mean=res_block_1/cnn_block_8/batch_normalization_8/moving_variancedense_2/kerneldense_2/bias*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**2
config_proto" 

CPU

GPU2 *0J 8 *+
f&R$
"__inference_signature_wrapper_8851
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ñ:
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp9res_block/cnn_block_3/conv2d_3/kernel/Read/ReadVariableOp7res_block/cnn_block_3/conv2d_3/bias/Read/ReadVariableOpEres_block/cnn_block_3/batch_normalization_3/gamma/Read/ReadVariableOpDres_block/cnn_block_3/batch_normalization_3/beta/Read/ReadVariableOp9res_block/cnn_block_4/conv2d_4/kernel/Read/ReadVariableOp7res_block/cnn_block_4/conv2d_4/bias/Read/ReadVariableOpEres_block/cnn_block_4/batch_normalization_4/gamma/Read/ReadVariableOpDres_block/cnn_block_4/batch_normalization_4/beta/Read/ReadVariableOp9res_block/cnn_block_5/conv2d_5/kernel/Read/ReadVariableOp7res_block/cnn_block_5/conv2d_5/bias/Read/ReadVariableOpEres_block/cnn_block_5/batch_normalization_5/gamma/Read/ReadVariableOpDres_block/cnn_block_5/batch_normalization_5/beta/Read/ReadVariableOp-res_block/conv2d_6/kernel/Read/ReadVariableOp+res_block/conv2d_6/bias/Read/ReadVariableOpKres_block/cnn_block_3/batch_normalization_3/moving_mean/Read/ReadVariableOpOres_block/cnn_block_3/batch_normalization_3/moving_variance/Read/ReadVariableOpKres_block/cnn_block_4/batch_normalization_4/moving_mean/Read/ReadVariableOpOres_block/cnn_block_4/batch_normalization_4/moving_variance/Read/ReadVariableOpKres_block/cnn_block_5/batch_normalization_5/moving_mean/Read/ReadVariableOpOres_block/cnn_block_5/batch_normalization_5/moving_variance/Read/ReadVariableOp;res_block_1/cnn_block_6/conv2d_7/kernel/Read/ReadVariableOp9res_block_1/cnn_block_6/conv2d_7/bias/Read/ReadVariableOpGres_block_1/cnn_block_6/batch_normalization_6/gamma/Read/ReadVariableOpFres_block_1/cnn_block_6/batch_normalization_6/beta/Read/ReadVariableOp;res_block_1/cnn_block_7/conv2d_8/kernel/Read/ReadVariableOp9res_block_1/cnn_block_7/conv2d_8/bias/Read/ReadVariableOpGres_block_1/cnn_block_7/batch_normalization_7/gamma/Read/ReadVariableOpFres_block_1/cnn_block_7/batch_normalization_7/beta/Read/ReadVariableOp;res_block_1/cnn_block_8/conv2d_9/kernel/Read/ReadVariableOp9res_block_1/cnn_block_8/conv2d_9/bias/Read/ReadVariableOpGres_block_1/cnn_block_8/batch_normalization_8/gamma/Read/ReadVariableOpFres_block_1/cnn_block_8/batch_normalization_8/beta/Read/ReadVariableOp0res_block_1/conv2d_10/kernel/Read/ReadVariableOp.res_block_1/conv2d_10/bias/Read/ReadVariableOpMres_block_1/cnn_block_6/batch_normalization_6/moving_mean/Read/ReadVariableOpQres_block_1/cnn_block_6/batch_normalization_6/moving_variance/Read/ReadVariableOpMres_block_1/cnn_block_7/batch_normalization_7/moving_mean/Read/ReadVariableOpQres_block_1/cnn_block_7/batch_normalization_7/moving_variance/Read/ReadVariableOpMres_block_1/cnn_block_8/batch_normalization_8/moving_mean/Read/ReadVariableOpQres_block_1/cnn_block_8/batch_normalization_8/moving_variance/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp@Adam/res_block/cnn_block_3/conv2d_3/kernel/m/Read/ReadVariableOp>Adam/res_block/cnn_block_3/conv2d_3/bias/m/Read/ReadVariableOpLAdam/res_block/cnn_block_3/batch_normalization_3/gamma/m/Read/ReadVariableOpKAdam/res_block/cnn_block_3/batch_normalization_3/beta/m/Read/ReadVariableOp@Adam/res_block/cnn_block_4/conv2d_4/kernel/m/Read/ReadVariableOp>Adam/res_block/cnn_block_4/conv2d_4/bias/m/Read/ReadVariableOpLAdam/res_block/cnn_block_4/batch_normalization_4/gamma/m/Read/ReadVariableOpKAdam/res_block/cnn_block_4/batch_normalization_4/beta/m/Read/ReadVariableOp@Adam/res_block/cnn_block_5/conv2d_5/kernel/m/Read/ReadVariableOp>Adam/res_block/cnn_block_5/conv2d_5/bias/m/Read/ReadVariableOpLAdam/res_block/cnn_block_5/batch_normalization_5/gamma/m/Read/ReadVariableOpKAdam/res_block/cnn_block_5/batch_normalization_5/beta/m/Read/ReadVariableOp4Adam/res_block/conv2d_6/kernel/m/Read/ReadVariableOp2Adam/res_block/conv2d_6/bias/m/Read/ReadVariableOpBAdam/res_block_1/cnn_block_6/conv2d_7/kernel/m/Read/ReadVariableOp@Adam/res_block_1/cnn_block_6/conv2d_7/bias/m/Read/ReadVariableOpNAdam/res_block_1/cnn_block_6/batch_normalization_6/gamma/m/Read/ReadVariableOpMAdam/res_block_1/cnn_block_6/batch_normalization_6/beta/m/Read/ReadVariableOpBAdam/res_block_1/cnn_block_7/conv2d_8/kernel/m/Read/ReadVariableOp@Adam/res_block_1/cnn_block_7/conv2d_8/bias/m/Read/ReadVariableOpNAdam/res_block_1/cnn_block_7/batch_normalization_7/gamma/m/Read/ReadVariableOpMAdam/res_block_1/cnn_block_7/batch_normalization_7/beta/m/Read/ReadVariableOpBAdam/res_block_1/cnn_block_8/conv2d_9/kernel/m/Read/ReadVariableOp@Adam/res_block_1/cnn_block_8/conv2d_9/bias/m/Read/ReadVariableOpNAdam/res_block_1/cnn_block_8/batch_normalization_8/gamma/m/Read/ReadVariableOpMAdam/res_block_1/cnn_block_8/batch_normalization_8/beta/m/Read/ReadVariableOp7Adam/res_block_1/conv2d_10/kernel/m/Read/ReadVariableOp5Adam/res_block_1/conv2d_10/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp@Adam/res_block/cnn_block_3/conv2d_3/kernel/v/Read/ReadVariableOp>Adam/res_block/cnn_block_3/conv2d_3/bias/v/Read/ReadVariableOpLAdam/res_block/cnn_block_3/batch_normalization_3/gamma/v/Read/ReadVariableOpKAdam/res_block/cnn_block_3/batch_normalization_3/beta/v/Read/ReadVariableOp@Adam/res_block/cnn_block_4/conv2d_4/kernel/v/Read/ReadVariableOp>Adam/res_block/cnn_block_4/conv2d_4/bias/v/Read/ReadVariableOpLAdam/res_block/cnn_block_4/batch_normalization_4/gamma/v/Read/ReadVariableOpKAdam/res_block/cnn_block_4/batch_normalization_4/beta/v/Read/ReadVariableOp@Adam/res_block/cnn_block_5/conv2d_5/kernel/v/Read/ReadVariableOp>Adam/res_block/cnn_block_5/conv2d_5/bias/v/Read/ReadVariableOpLAdam/res_block/cnn_block_5/batch_normalization_5/gamma/v/Read/ReadVariableOpKAdam/res_block/cnn_block_5/batch_normalization_5/beta/v/Read/ReadVariableOp4Adam/res_block/conv2d_6/kernel/v/Read/ReadVariableOp2Adam/res_block/conv2d_6/bias/v/Read/ReadVariableOpBAdam/res_block_1/cnn_block_6/conv2d_7/kernel/v/Read/ReadVariableOp@Adam/res_block_1/cnn_block_6/conv2d_7/bias/v/Read/ReadVariableOpNAdam/res_block_1/cnn_block_6/batch_normalization_6/gamma/v/Read/ReadVariableOpMAdam/res_block_1/cnn_block_6/batch_normalization_6/beta/v/Read/ReadVariableOpBAdam/res_block_1/cnn_block_7/conv2d_8/kernel/v/Read/ReadVariableOp@Adam/res_block_1/cnn_block_7/conv2d_8/bias/v/Read/ReadVariableOpNAdam/res_block_1/cnn_block_7/batch_normalization_7/gamma/v/Read/ReadVariableOpMAdam/res_block_1/cnn_block_7/batch_normalization_7/beta/v/Read/ReadVariableOpBAdam/res_block_1/cnn_block_8/conv2d_9/kernel/v/Read/ReadVariableOp@Adam/res_block_1/cnn_block_8/conv2d_9/bias/v/Read/ReadVariableOpNAdam/res_block_1/cnn_block_8/batch_normalization_8/gamma/v/Read/ReadVariableOpMAdam/res_block_1/cnn_block_8/batch_normalization_8/beta/v/Read/ReadVariableOp7Adam/res_block_1/conv2d_10/kernel/v/Read/ReadVariableOp5Adam/res_block_1/conv2d_10/bias/v/Read/ReadVariableOpConst*|
Tinu
s2q	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *'
f"R 
__inference__traced_save_10595
À)
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/bias%res_block/cnn_block_3/conv2d_3/kernel#res_block/cnn_block_3/conv2d_3/bias1res_block/cnn_block_3/batch_normalization_3/gamma0res_block/cnn_block_3/batch_normalization_3/beta%res_block/cnn_block_4/conv2d_4/kernel#res_block/cnn_block_4/conv2d_4/bias1res_block/cnn_block_4/batch_normalization_4/gamma0res_block/cnn_block_4/batch_normalization_4/beta%res_block/cnn_block_5/conv2d_5/kernel#res_block/cnn_block_5/conv2d_5/bias1res_block/cnn_block_5/batch_normalization_5/gamma0res_block/cnn_block_5/batch_normalization_5/betares_block/conv2d_6/kernelres_block/conv2d_6/bias7res_block/cnn_block_3/batch_normalization_3/moving_mean;res_block/cnn_block_3/batch_normalization_3/moving_variance7res_block/cnn_block_4/batch_normalization_4/moving_mean;res_block/cnn_block_4/batch_normalization_4/moving_variance7res_block/cnn_block_5/batch_normalization_5/moving_mean;res_block/cnn_block_5/batch_normalization_5/moving_variance'res_block_1/cnn_block_6/conv2d_7/kernel%res_block_1/cnn_block_6/conv2d_7/bias3res_block_1/cnn_block_6/batch_normalization_6/gamma2res_block_1/cnn_block_6/batch_normalization_6/beta'res_block_1/cnn_block_7/conv2d_8/kernel%res_block_1/cnn_block_7/conv2d_8/bias3res_block_1/cnn_block_7/batch_normalization_7/gamma2res_block_1/cnn_block_7/batch_normalization_7/beta'res_block_1/cnn_block_8/conv2d_9/kernel%res_block_1/cnn_block_8/conv2d_9/bias3res_block_1/cnn_block_8/batch_normalization_8/gamma2res_block_1/cnn_block_8/batch_normalization_8/betares_block_1/conv2d_10/kernelres_block_1/conv2d_10/bias9res_block_1/cnn_block_6/batch_normalization_6/moving_mean=res_block_1/cnn_block_6/batch_normalization_6/moving_variance9res_block_1/cnn_block_7/batch_normalization_7/moving_mean=res_block_1/cnn_block_7/batch_normalization_7/moving_variance9res_block_1/cnn_block_8/batch_normalization_8/moving_mean=res_block_1/cnn_block_8/batch_normalization_8/moving_variance	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_2/kernel/mAdam/dense_2/bias/m,Adam/res_block/cnn_block_3/conv2d_3/kernel/m*Adam/res_block/cnn_block_3/conv2d_3/bias/m8Adam/res_block/cnn_block_3/batch_normalization_3/gamma/m7Adam/res_block/cnn_block_3/batch_normalization_3/beta/m,Adam/res_block/cnn_block_4/conv2d_4/kernel/m*Adam/res_block/cnn_block_4/conv2d_4/bias/m8Adam/res_block/cnn_block_4/batch_normalization_4/gamma/m7Adam/res_block/cnn_block_4/batch_normalization_4/beta/m,Adam/res_block/cnn_block_5/conv2d_5/kernel/m*Adam/res_block/cnn_block_5/conv2d_5/bias/m8Adam/res_block/cnn_block_5/batch_normalization_5/gamma/m7Adam/res_block/cnn_block_5/batch_normalization_5/beta/m Adam/res_block/conv2d_6/kernel/mAdam/res_block/conv2d_6/bias/m.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/m,Adam/res_block_1/cnn_block_6/conv2d_7/bias/m:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/m9Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/m.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/m,Adam/res_block_1/cnn_block_7/conv2d_8/bias/m:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/m9Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/m.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/m,Adam/res_block_1/cnn_block_8/conv2d_9/bias/m:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/m9Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/m#Adam/res_block_1/conv2d_10/kernel/m!Adam/res_block_1/conv2d_10/bias/mAdam/dense_2/kernel/vAdam/dense_2/bias/v,Adam/res_block/cnn_block_3/conv2d_3/kernel/v*Adam/res_block/cnn_block_3/conv2d_3/bias/v8Adam/res_block/cnn_block_3/batch_normalization_3/gamma/v7Adam/res_block/cnn_block_3/batch_normalization_3/beta/v,Adam/res_block/cnn_block_4/conv2d_4/kernel/v*Adam/res_block/cnn_block_4/conv2d_4/bias/v8Adam/res_block/cnn_block_4/batch_normalization_4/gamma/v7Adam/res_block/cnn_block_4/batch_normalization_4/beta/v,Adam/res_block/cnn_block_5/conv2d_5/kernel/v*Adam/res_block/cnn_block_5/conv2d_5/bias/v8Adam/res_block/cnn_block_5/batch_normalization_5/gamma/v7Adam/res_block/cnn_block_5/batch_normalization_5/beta/v Adam/res_block/conv2d_6/kernel/vAdam/res_block/conv2d_6/bias/v.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/v,Adam/res_block_1/cnn_block_6/conv2d_7/bias/v:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/v9Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/v.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/v,Adam/res_block_1/cnn_block_7/conv2d_8/bias/v:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/v9Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/v.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/v,Adam/res_block_1/cnn_block_8/conv2d_9/bias/v:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/v9Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/v#Adam/res_block_1/conv2d_10/kernel/v!Adam/res_block_1/conv2d_10/bias/v*{
Tint
r2p*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 **
f%R#
!__inference__traced_restore_10938·´
k
¤
C__inference_res_block_layer_call_and_return_conditional_losses_9502
input_tensorM
3cnn_block_3_conv2d_3_conv2d_readvariableop_resource: B
4cnn_block_3_conv2d_3_biasadd_readvariableop_resource: G
9cnn_block_3_batch_normalization_3_readvariableop_resource: I
;cnn_block_3_batch_normalization_3_readvariableop_1_resource: X
Jcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource: Z
Lcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: M
3cnn_block_4_conv2d_4_conv2d_readvariableop_resource:  B
4cnn_block_4_conv2d_4_biasadd_readvariableop_resource: G
9cnn_block_4_batch_normalization_4_readvariableop_resource: I
;cnn_block_4_batch_normalization_4_readvariableop_1_resource: X
Jcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource: Z
Lcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: M
3cnn_block_5_conv2d_5_conv2d_readvariableop_resource: @B
4cnn_block_5_conv2d_5_biasadd_readvariableop_resource:@G
9cnn_block_5_batch_normalization_5_readvariableop_resource:@I
;cnn_block_5_batch_normalization_5_readvariableop_1_resource:@X
Jcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@Z
Lcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@
identity¢Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_3/batch_normalization_3/ReadVariableOp¢2cnn_block_3/batch_normalization_3/ReadVariableOp_1¢+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp¢*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp¢Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_4/batch_normalization_4/ReadVariableOp¢2cnn_block_4/batch_normalization_4/ReadVariableOp_1¢+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp¢*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp¢Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_5/batch_normalization_5/ReadVariableOp¢2cnn_block_5/batch_normalization_5/ReadVariableOp_1¢+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp¢*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp¢conv2d_6/BiasAdd/ReadVariableOp¢conv2d_6/Conv2D/ReadVariableOp¦
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0É
cnn_block_3/conv2d_3/Conv2DConv2Dinput_tensor2cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¼
cnn_block_3/conv2d_3/BiasAddBiasAdd$cnn_block_3/conv2d_3/Conv2D:output:03cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
0cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOp9cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype0ª
2cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp;cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ì
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ÿ
2cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%cnn_block_3/conv2d_3/BiasAdd:output:08cnn_block_3/batch_normalization_3/ReadVariableOp:value:0:cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0Icnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 
cnn_block_3/ReluRelu6cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Û
cnn_block_4/conv2d_4/Conv2DConv2Dcnn_block_3/Relu:activations:02cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¼
cnn_block_4/conv2d_4/BiasAddBiasAdd$cnn_block_4/conv2d_4/Conv2D:output:03cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
0cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOp9cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype0ª
2cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp;cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype0È
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ì
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ÿ
2cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%cnn_block_4/conv2d_4/BiasAdd:output:08cnn_block_4/batch_normalization_4/ReadVariableOp:value:0:cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0Icnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 
cnn_block_4/ReluRelu6cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0±
conv2d_6/Conv2DConv2Dinput_tensor&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
addAddV2cnn_block_4/Relu:activations:0conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ä
cnn_block_5/conv2d_5/Conv2DConv2Dadd:z:02cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¼
cnn_block_5/conv2d_5/BiasAddBiasAdd$cnn_block_5/conv2d_5/Conv2D:output:03cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
0cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOp9cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0ª
2cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp;cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ì
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ÿ
2cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%cnn_block_5/conv2d_5/BiasAdd:output:08cnn_block_5/batch_normalization_5/ReadVariableOp:value:0:cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0Icnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 
cnn_block_5/ReluRelu6cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@­
max_pooling2d/MaxPoolMaxPoolcnn_block_5/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
u
IdentityIdentitymax_pooling2d/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ð
NoOpNoOpB^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpD^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_11^cnn_block_3/batch_normalization_3/ReadVariableOp3^cnn_block_3/batch_normalization_3/ReadVariableOp_1,^cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp+^cnn_block_3/conv2d_3/Conv2D/ReadVariableOpB^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpD^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_11^cnn_block_4/batch_normalization_4/ReadVariableOp3^cnn_block_4/batch_normalization_4/ReadVariableOp_1,^cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp+^cnn_block_4/conv2d_4/Conv2D/ReadVariableOpB^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpD^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_11^cnn_block_5/batch_normalization_5/ReadVariableOp3^cnn_block_5/batch_normalization_5/ReadVariableOp_1,^cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp+^cnn_block_5/conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpAcnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_3/batch_normalization_3/ReadVariableOp0cnn_block_3/batch_normalization_3/ReadVariableOp2h
2cnn_block_3/batch_normalization_3/ReadVariableOp_12cnn_block_3/batch_normalization_3/ReadVariableOp_12Z
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp2X
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp2
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpAcnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_4/batch_normalization_4/ReadVariableOp0cnn_block_4/batch_normalization_4/ReadVariableOp2h
2cnn_block_4/batch_normalization_4/ReadVariableOp_12cnn_block_4/batch_normalization_4/ReadVariableOp_12Z
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp2X
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp2
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpAcnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_5/batch_normalization_5/ReadVariableOp0cnn_block_5/batch_normalization_5/ReadVariableOp2h
2cnn_block_5/batch_normalization_5/ReadVariableOp_12cnn_block_5/batch_normalization_5/ReadVariableOp_12Z
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp2X
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_tensor

Ü
C__inference_res_block_layer_call_and_return_conditional_losses_8168
input_tensorM
3cnn_block_3_conv2d_3_conv2d_readvariableop_resource: B
4cnn_block_3_conv2d_3_biasadd_readvariableop_resource: G
9cnn_block_3_batch_normalization_3_readvariableop_resource: I
;cnn_block_3_batch_normalization_3_readvariableop_1_resource: X
Jcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource: Z
Lcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: M
3cnn_block_4_conv2d_4_conv2d_readvariableop_resource:  B
4cnn_block_4_conv2d_4_biasadd_readvariableop_resource: G
9cnn_block_4_batch_normalization_4_readvariableop_resource: I
;cnn_block_4_batch_normalization_4_readvariableop_1_resource: X
Jcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource: Z
Lcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: M
3cnn_block_5_conv2d_5_conv2d_readvariableop_resource: @B
4cnn_block_5_conv2d_5_biasadd_readvariableop_resource:@G
9cnn_block_5_batch_normalization_5_readvariableop_resource:@I
;cnn_block_5_batch_normalization_5_readvariableop_1_resource:@X
Jcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@Z
Lcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@
identity¢0cnn_block_3/batch_normalization_3/AssignNewValue¢2cnn_block_3/batch_normalization_3/AssignNewValue_1¢Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_3/batch_normalization_3/ReadVariableOp¢2cnn_block_3/batch_normalization_3/ReadVariableOp_1¢+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp¢*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp¢0cnn_block_4/batch_normalization_4/AssignNewValue¢2cnn_block_4/batch_normalization_4/AssignNewValue_1¢Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_4/batch_normalization_4/ReadVariableOp¢2cnn_block_4/batch_normalization_4/ReadVariableOp_1¢+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp¢*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp¢0cnn_block_5/batch_normalization_5/AssignNewValue¢2cnn_block_5/batch_normalization_5/AssignNewValue_1¢Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_5/batch_normalization_5/ReadVariableOp¢2cnn_block_5/batch_normalization_5/ReadVariableOp_1¢+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp¢*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp¢conv2d_6/BiasAdd/ReadVariableOp¢conv2d_6/Conv2D/ReadVariableOp¦
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0É
cnn_block_3/conv2d_3/Conv2DConv2Dinput_tensor2cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¼
cnn_block_3/conv2d_3/BiasAddBiasAdd$cnn_block_3/conv2d_3/Conv2D:output:03cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
0cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOp9cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype0ª
2cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp;cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ì
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0
2cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%cnn_block_3/conv2d_3/BiasAdd:output:08cnn_block_3/batch_normalization_3/ReadVariableOp:value:0:cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0Icnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<Î
0cnn_block_3/batch_normalization_3/AssignNewValueAssignVariableOpJcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource?cnn_block_3/batch_normalization_3/FusedBatchNormV3:batch_mean:0B^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ø
2cnn_block_3/batch_normalization_3/AssignNewValue_1AssignVariableOpLcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_3/batch_normalization_3/FusedBatchNormV3:batch_variance:0D^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
cnn_block_3/ReluRelu6cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Û
cnn_block_4/conv2d_4/Conv2DConv2Dcnn_block_3/Relu:activations:02cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¼
cnn_block_4/conv2d_4/BiasAddBiasAdd$cnn_block_4/conv2d_4/Conv2D:output:03cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
0cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOp9cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype0ª
2cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp;cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype0È
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ì
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0
2cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%cnn_block_4/conv2d_4/BiasAdd:output:08cnn_block_4/batch_normalization_4/ReadVariableOp:value:0:cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0Icnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<Î
0cnn_block_4/batch_normalization_4/AssignNewValueAssignVariableOpJcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource?cnn_block_4/batch_normalization_4/FusedBatchNormV3:batch_mean:0B^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ø
2cnn_block_4/batch_normalization_4/AssignNewValue_1AssignVariableOpLcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_4/batch_normalization_4/FusedBatchNormV3:batch_variance:0D^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
cnn_block_4/ReluRelu6cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0±
conv2d_6/Conv2DConv2Dinput_tensor&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
addAddV2cnn_block_4/Relu:activations:0conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ä
cnn_block_5/conv2d_5/Conv2DConv2Dadd:z:02cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¼
cnn_block_5/conv2d_5/BiasAddBiasAdd$cnn_block_5/conv2d_5/Conv2D:output:03cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
0cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOp9cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0ª
2cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp;cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ì
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
2cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%cnn_block_5/conv2d_5/BiasAdd:output:08cnn_block_5/batch_normalization_5/ReadVariableOp:value:0:cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0Icnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<Î
0cnn_block_5/batch_normalization_5/AssignNewValueAssignVariableOpJcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource?cnn_block_5/batch_normalization_5/FusedBatchNormV3:batch_mean:0B^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ø
2cnn_block_5/batch_normalization_5/AssignNewValue_1AssignVariableOpLcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_5/batch_normalization_5/FusedBatchNormV3:batch_variance:0D^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
cnn_block_5/ReluRelu6cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@­
max_pooling2d/MaxPoolMaxPoolcnn_block_5/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
u
IdentityIdentitymax_pooling2d/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¨
NoOpNoOp1^cnn_block_3/batch_normalization_3/AssignNewValue3^cnn_block_3/batch_normalization_3/AssignNewValue_1B^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpD^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_11^cnn_block_3/batch_normalization_3/ReadVariableOp3^cnn_block_3/batch_normalization_3/ReadVariableOp_1,^cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp+^cnn_block_3/conv2d_3/Conv2D/ReadVariableOp1^cnn_block_4/batch_normalization_4/AssignNewValue3^cnn_block_4/batch_normalization_4/AssignNewValue_1B^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpD^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_11^cnn_block_4/batch_normalization_4/ReadVariableOp3^cnn_block_4/batch_normalization_4/ReadVariableOp_1,^cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp+^cnn_block_4/conv2d_4/Conv2D/ReadVariableOp1^cnn_block_5/batch_normalization_5/AssignNewValue3^cnn_block_5/batch_normalization_5/AssignNewValue_1B^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpD^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_11^cnn_block_5/batch_normalization_5/ReadVariableOp3^cnn_block_5/batch_normalization_5/ReadVariableOp_1,^cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp+^cnn_block_5/conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2d
0cnn_block_3/batch_normalization_3/AssignNewValue0cnn_block_3/batch_normalization_3/AssignNewValue2h
2cnn_block_3/batch_normalization_3/AssignNewValue_12cnn_block_3/batch_normalization_3/AssignNewValue_12
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpAcnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_3/batch_normalization_3/ReadVariableOp0cnn_block_3/batch_normalization_3/ReadVariableOp2h
2cnn_block_3/batch_normalization_3/ReadVariableOp_12cnn_block_3/batch_normalization_3/ReadVariableOp_12Z
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp2X
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp2d
0cnn_block_4/batch_normalization_4/AssignNewValue0cnn_block_4/batch_normalization_4/AssignNewValue2h
2cnn_block_4/batch_normalization_4/AssignNewValue_12cnn_block_4/batch_normalization_4/AssignNewValue_12
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpAcnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_4/batch_normalization_4/ReadVariableOp0cnn_block_4/batch_normalization_4/ReadVariableOp2h
2cnn_block_4/batch_normalization_4/ReadVariableOp_12cnn_block_4/batch_normalization_4/ReadVariableOp_12Z
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp2X
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp2d
0cnn_block_5/batch_normalization_5/AssignNewValue0cnn_block_5/batch_normalization_5/AssignNewValue2h
2cnn_block_5/batch_normalization_5/AssignNewValue_12cnn_block_5/batch_normalization_5/AssignNewValue_12
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpAcnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_5/batch_normalization_5/ReadVariableOp0cnn_block_5/batch_normalization_5/ReadVariableOp2h
2cnn_block_5/batch_normalization_5/ReadVariableOp_12cnn_block_5/batch_normalization_5/ReadVariableOp_12Z
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp2X
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_tensor

¾
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_7219

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
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
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

¾
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9991

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
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
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
ú
ø
E__inference_res_block_1_layer_call_and_return_conditional_losses_9817
input_tensorN
3cnn_block_6_conv2d_7_conv2d_readvariableop_resource:@C
4cnn_block_6_conv2d_7_biasadd_readvariableop_resource:	H
9cnn_block_6_batch_normalization_6_readvariableop_resource:	J
;cnn_block_6_batch_normalization_6_readvariableop_1_resource:	Y
Jcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	[
Lcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	O
3cnn_block_7_conv2d_8_conv2d_readvariableop_resource:C
4cnn_block_7_conv2d_8_biasadd_readvariableop_resource:	H
9cnn_block_7_batch_normalization_7_readvariableop_resource:	J
;cnn_block_7_batch_normalization_7_readvariableop_1_resource:	Y
Jcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	[
Lcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	C
(conv2d_10_conv2d_readvariableop_resource:@8
)conv2d_10_biasadd_readvariableop_resource:	O
3cnn_block_8_conv2d_9_conv2d_readvariableop_resource:C
4cnn_block_8_conv2d_9_biasadd_readvariableop_resource:	H
9cnn_block_8_batch_normalization_8_readvariableop_resource:	J
;cnn_block_8_batch_normalization_8_readvariableop_1_resource:	Y
Jcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	[
Lcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	
identity¢0cnn_block_6/batch_normalization_6/AssignNewValue¢2cnn_block_6/batch_normalization_6/AssignNewValue_1¢Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_6/batch_normalization_6/ReadVariableOp¢2cnn_block_6/batch_normalization_6/ReadVariableOp_1¢+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp¢*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp¢0cnn_block_7/batch_normalization_7/AssignNewValue¢2cnn_block_7/batch_normalization_7/AssignNewValue_1¢Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_7/batch_normalization_7/ReadVariableOp¢2cnn_block_7/batch_normalization_7/ReadVariableOp_1¢+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp¢*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp¢0cnn_block_8/batch_normalization_8/AssignNewValue¢2cnn_block_8/batch_normalization_8/AssignNewValue_1¢Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_8/batch_normalization_8/ReadVariableOp¢2cnn_block_8/batch_normalization_8/ReadVariableOp_1¢+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp¢*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp¢ conv2d_10/BiasAdd/ReadVariableOp¢conv2d_10/Conv2D/ReadVariableOp§
*cnn_block_6/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3cnn_block_6_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ê
cnn_block_6/conv2d_7/Conv2DConv2Dinput_tensor2cnn_block_6/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_6_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
cnn_block_6/conv2d_7/BiasAddBiasAdd$cnn_block_6/conv2d_7/Conv2D:output:03cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOp9cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype0«
2cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp;cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype0É
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Í
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
2cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%cnn_block_6/conv2d_7/BiasAdd:output:08cnn_block_6/batch_normalization_6/ReadVariableOp:value:0:cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0Icnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Î
0cnn_block_6/batch_normalization_6/AssignNewValueAssignVariableOpJcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource?cnn_block_6/batch_normalization_6/FusedBatchNormV3:batch_mean:0B^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ø
2cnn_block_6/batch_normalization_6/AssignNewValue_1AssignVariableOpLcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_6/batch_normalization_6/FusedBatchNormV3:batch_variance:0D^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
cnn_block_6/ReluRelu6cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
*cnn_block_7/conv2d_8/Conv2D/ReadVariableOpReadVariableOp3cnn_block_7_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
cnn_block_7/conv2d_8/Conv2DConv2Dcnn_block_6/Relu:activations:02cnn_block_7/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_7_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
cnn_block_7/conv2d_8/BiasAddBiasAdd$cnn_block_7/conv2d_8/Conv2D:output:03cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0cnn_block_7/batch_normalization_7/ReadVariableOpReadVariableOp9cnn_block_7_batch_normalization_7_readvariableop_resource*
_output_shapes	
:*
dtype0«
2cnn_block_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp;cnn_block_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:*
dtype0É
Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Í
Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
2cnn_block_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%cnn_block_7/conv2d_8/BiasAdd:output:08cnn_block_7/batch_normalization_7/ReadVariableOp:value:0:cnn_block_7/batch_normalization_7/ReadVariableOp_1:value:0Icnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Î
0cnn_block_7/batch_normalization_7/AssignNewValueAssignVariableOpJcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource?cnn_block_7/batch_normalization_7/FusedBatchNormV3:batch_mean:0B^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ø
2cnn_block_7/batch_normalization_7/AssignNewValue_1AssignVariableOpLcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_7/batch_normalization_7/FusedBatchNormV3:batch_variance:0D^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
cnn_block_7/ReluRelu6cnn_block_7/batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0´
conv2d_10/Conv2DConv2Dinput_tensor'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
addAddV2cnn_block_7/Relu:activations:0conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
*cnn_block_8/conv2d_9/Conv2D/ReadVariableOpReadVariableOp3cnn_block_8_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Å
cnn_block_8/conv2d_9/Conv2DConv2Dadd:z:02cnn_block_8/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_8_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
cnn_block_8/conv2d_9/BiasAddBiasAdd$cnn_block_8/conv2d_9/Conv2D:output:03cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0cnn_block_8/batch_normalization_8/ReadVariableOpReadVariableOp9cnn_block_8_batch_normalization_8_readvariableop_resource*
_output_shapes	
:*
dtype0«
2cnn_block_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp;cnn_block_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:*
dtype0É
Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Í
Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
2cnn_block_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3%cnn_block_8/conv2d_9/BiasAdd:output:08cnn_block_8/batch_normalization_8/ReadVariableOp:value:0:cnn_block_8/batch_normalization_8/ReadVariableOp_1:value:0Icnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Î
0cnn_block_8/batch_normalization_8/AssignNewValueAssignVariableOpJcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource?cnn_block_8/batch_normalization_8/FusedBatchNormV3:batch_mean:0B^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ø
2cnn_block_8/batch_normalization_8/AssignNewValue_1AssignVariableOpLcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_8/batch_normalization_8/FusedBatchNormV3:batch_variance:0D^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
cnn_block_8/ReluRelu6cnn_block_8/batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
max_pooling2d_1/MaxPoolMaxPoolcnn_block_8/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
x
IdentityIdentity max_pooling2d_1/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
NoOpNoOp1^cnn_block_6/batch_normalization_6/AssignNewValue3^cnn_block_6/batch_normalization_6/AssignNewValue_1B^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpD^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_11^cnn_block_6/batch_normalization_6/ReadVariableOp3^cnn_block_6/batch_normalization_6/ReadVariableOp_1,^cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp+^cnn_block_6/conv2d_7/Conv2D/ReadVariableOp1^cnn_block_7/batch_normalization_7/AssignNewValue3^cnn_block_7/batch_normalization_7/AssignNewValue_1B^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpD^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_11^cnn_block_7/batch_normalization_7/ReadVariableOp3^cnn_block_7/batch_normalization_7/ReadVariableOp_1,^cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp+^cnn_block_7/conv2d_8/Conv2D/ReadVariableOp1^cnn_block_8/batch_normalization_8/AssignNewValue3^cnn_block_8/batch_normalization_8/AssignNewValue_1B^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpD^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_11^cnn_block_8/batch_normalization_8/ReadVariableOp3^cnn_block_8/batch_normalization_8/ReadVariableOp_1,^cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp+^cnn_block_8/conv2d_9/Conv2D/ReadVariableOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ@: : : : : : : : : : : : : : : : : : : : 2d
0cnn_block_6/batch_normalization_6/AssignNewValue0cnn_block_6/batch_normalization_6/AssignNewValue2h
2cnn_block_6/batch_normalization_6/AssignNewValue_12cnn_block_6/batch_normalization_6/AssignNewValue_12
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpAcnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_6/batch_normalization_6/ReadVariableOp0cnn_block_6/batch_normalization_6/ReadVariableOp2h
2cnn_block_6/batch_normalization_6/ReadVariableOp_12cnn_block_6/batch_normalization_6/ReadVariableOp_12Z
+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp2X
*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp2d
0cnn_block_7/batch_normalization_7/AssignNewValue0cnn_block_7/batch_normalization_7/AssignNewValue2h
2cnn_block_7/batch_normalization_7/AssignNewValue_12cnn_block_7/batch_normalization_7/AssignNewValue_12
Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpAcnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_7/batch_normalization_7/ReadVariableOp0cnn_block_7/batch_normalization_7/ReadVariableOp2h
2cnn_block_7/batch_normalization_7/ReadVariableOp_12cnn_block_7/batch_normalization_7/ReadVariableOp_12Z
+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp2X
*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp2d
0cnn_block_8/batch_normalization_8/AssignNewValue0cnn_block_8/batch_normalization_8/AssignNewValue2h
2cnn_block_8/batch_normalization_8/AssignNewValue_12cnn_block_8/batch_normalization_8/AssignNewValue_12
Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpAcnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_8/batch_normalization_8/ReadVariableOp0cnn_block_8/batch_normalization_8/ReadVariableOp2h
2cnn_block_8/batch_normalization_8/ReadVariableOp_12cnn_block_8/batch_normalization_8/ReadVariableOp_12Z
+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp2X
*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
&
_user_specified_nameinput_tensor
Ä

&__inference_dense_2_layer_call_fn_9837

inputs
unknown:	b

	unknown_0:

identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_7769o
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
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿb: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 
_user_specified_nameinputs

ª
*__inference_res_block_1_layer_call_fn_9667
input_tensor"
unknown:@
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	%

unknown_11:@

unknown_12:	&

unknown_13:

unknown_14:	

unknown_15:	

unknown_16:	

unknown_17:	

unknown_18:	
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_res_block_1_layer_call_and_return_conditional_losses_8002x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ@: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
&
_user_specified_nameinput_tensor
È	
ó
A__inference_dense_2_layer_call_and_return_conditional_losses_9847

inputs1
matmul_readvariableop_resource:	b
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	b
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿb: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 
_user_specified_nameinputs
Û

P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10159

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
Ú

O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7392

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
Ú

O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7328

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

¾
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_7155

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
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
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
	
Ï
4__inference_batch_normalization_4_layer_call_fn_9955

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
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
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_7219
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
ÿ

(__inference_res_block_layer_call_fn_9427
input_tensor!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: $

unknown_13: @

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_res_block_layer_call_and_return_conditional_losses_8168w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_tensor

Â
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7359

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
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
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

Ã
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10239

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È
7
__inference__wrapped_model_7102
input_1_
Emodel_1_res_block_cnn_block_3_conv2d_3_conv2d_readvariableop_resource: T
Fmodel_1_res_block_cnn_block_3_conv2d_3_biasadd_readvariableop_resource: Y
Kmodel_1_res_block_cnn_block_3_batch_normalization_3_readvariableop_resource: [
Mmodel_1_res_block_cnn_block_3_batch_normalization_3_readvariableop_1_resource: j
\model_1_res_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource: l
^model_1_res_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: _
Emodel_1_res_block_cnn_block_4_conv2d_4_conv2d_readvariableop_resource:  T
Fmodel_1_res_block_cnn_block_4_conv2d_4_biasadd_readvariableop_resource: Y
Kmodel_1_res_block_cnn_block_4_batch_normalization_4_readvariableop_resource: [
Mmodel_1_res_block_cnn_block_4_batch_normalization_4_readvariableop_1_resource: j
\model_1_res_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource: l
^model_1_res_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: S
9model_1_res_block_conv2d_6_conv2d_readvariableop_resource: H
:model_1_res_block_conv2d_6_biasadd_readvariableop_resource: _
Emodel_1_res_block_cnn_block_5_conv2d_5_conv2d_readvariableop_resource: @T
Fmodel_1_res_block_cnn_block_5_conv2d_5_biasadd_readvariableop_resource:@Y
Kmodel_1_res_block_cnn_block_5_batch_normalization_5_readvariableop_resource:@[
Mmodel_1_res_block_cnn_block_5_batch_normalization_5_readvariableop_1_resource:@j
\model_1_res_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@l
^model_1_res_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@b
Gmodel_1_res_block_1_cnn_block_6_conv2d_7_conv2d_readvariableop_resource:@W
Hmodel_1_res_block_1_cnn_block_6_conv2d_7_biasadd_readvariableop_resource:	\
Mmodel_1_res_block_1_cnn_block_6_batch_normalization_6_readvariableop_resource:	^
Omodel_1_res_block_1_cnn_block_6_batch_normalization_6_readvariableop_1_resource:	m
^model_1_res_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	o
`model_1_res_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	c
Gmodel_1_res_block_1_cnn_block_7_conv2d_8_conv2d_readvariableop_resource:W
Hmodel_1_res_block_1_cnn_block_7_conv2d_8_biasadd_readvariableop_resource:	\
Mmodel_1_res_block_1_cnn_block_7_batch_normalization_7_readvariableop_resource:	^
Omodel_1_res_block_1_cnn_block_7_batch_normalization_7_readvariableop_1_resource:	m
^model_1_res_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	o
`model_1_res_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	W
<model_1_res_block_1_conv2d_10_conv2d_readvariableop_resource:@L
=model_1_res_block_1_conv2d_10_biasadd_readvariableop_resource:	c
Gmodel_1_res_block_1_cnn_block_8_conv2d_9_conv2d_readvariableop_resource:W
Hmodel_1_res_block_1_cnn_block_8_conv2d_9_biasadd_readvariableop_resource:	\
Mmodel_1_res_block_1_cnn_block_8_batch_normalization_8_readvariableop_resource:	^
Omodel_1_res_block_1_cnn_block_8_batch_normalization_8_readvariableop_1_resource:	m
^model_1_res_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	o
`model_1_res_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	A
.model_1_dense_2_matmul_readvariableop_resource:	b
=
/model_1_dense_2_biasadd_readvariableop_resource:

identity¢&model_1/dense_2/BiasAdd/ReadVariableOp¢%model_1/dense_2/MatMul/ReadVariableOp¢Smodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢Umodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢Bmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOp¢Dmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1¢=model_1/res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp¢<model_1/res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp¢Smodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp¢Umodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1¢Bmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOp¢Dmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1¢=model_1/res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp¢<model_1/res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp¢Smodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp¢Umodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1¢Bmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOp¢Dmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1¢=model_1/res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp¢<model_1/res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp¢1model_1/res_block/conv2d_6/BiasAdd/ReadVariableOp¢0model_1/res_block/conv2d_6/Conv2D/ReadVariableOp¢Umodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp¢Wmodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1¢Dmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp¢Fmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1¢?model_1/res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp¢>model_1/res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp¢Umodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp¢Wmodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1¢Dmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp¢Fmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1¢?model_1/res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp¢>model_1/res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp¢Umodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢Wmodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢Dmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp¢Fmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1¢?model_1/res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp¢>model_1/res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp¢4model_1/res_block_1/conv2d_10/BiasAdd/ReadVariableOp¢3model_1/res_block_1/conv2d_10/Conv2D/ReadVariableOpÊ
<model_1/res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOpEmodel_1_res_block_cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0è
-model_1/res_block/cnn_block_3/conv2d_3/Conv2DConv2Dinput_1Dmodel_1/res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
À
=model_1/res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpFmodel_1_res_block_cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ò
.model_1/res_block/cnn_block_3/conv2d_3/BiasAddBiasAdd6model_1/res_block/cnn_block_3/conv2d_3/Conv2D:output:0Emodel_1/res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ê
Bmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOpKmodel_1_res_block_cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype0Î
Dmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOpMmodel_1_res_block_cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype0ì
Smodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp\model_1_res_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ð
Umodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp^model_1_res_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ë
Dmodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV37model_1/res_block/cnn_block_3/conv2d_3/BiasAdd:output:0Jmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOp:value:0Lmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0[model_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0]model_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( ®
"model_1/res_block/cnn_block_3/ReluReluHmodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ê
<model_1/res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOpEmodel_1_res_block_cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0
-model_1/res_block/cnn_block_4/conv2d_4/Conv2DConv2D0model_1/res_block/cnn_block_3/Relu:activations:0Dmodel_1/res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
À
=model_1/res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpFmodel_1_res_block_cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ò
.model_1/res_block/cnn_block_4/conv2d_4/BiasAddBiasAdd6model_1/res_block/cnn_block_4/conv2d_4/Conv2D:output:0Emodel_1/res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ê
Bmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOpKmodel_1_res_block_cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype0Î
Dmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOpMmodel_1_res_block_cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype0ì
Smodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp\model_1_res_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ð
Umodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp^model_1_res_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ë
Dmodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV37model_1/res_block/cnn_block_4/conv2d_4/BiasAdd:output:0Jmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOp:value:0Lmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0[model_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0]model_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( ®
"model_1/res_block/cnn_block_4/ReluReluHmodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ²
0model_1/res_block/conv2d_6/Conv2D/ReadVariableOpReadVariableOp9model_1_res_block_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ð
!model_1/res_block/conv2d_6/Conv2DConv2Dinput_18model_1/res_block/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¨
1model_1/res_block/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp:model_1_res_block_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Î
"model_1/res_block/conv2d_6/BiasAddBiasAdd*model_1/res_block/conv2d_6/Conv2D:output:09model_1/res_block/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ·
model_1/res_block/addAddV20model_1/res_block/cnn_block_4/Relu:activations:0+model_1/res_block/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ê
<model_1/res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOpEmodel_1_res_block_cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0ú
-model_1/res_block/cnn_block_5/conv2d_5/Conv2DConv2Dmodel_1/res_block/add:z:0Dmodel_1/res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
À
=model_1/res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpFmodel_1_res_block_cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ò
.model_1/res_block/cnn_block_5/conv2d_5/BiasAddBiasAdd6model_1/res_block/cnn_block_5/conv2d_5/Conv2D:output:0Emodel_1/res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
Bmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOpKmodel_1_res_block_cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0Î
Dmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOpMmodel_1_res_block_cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0ì
Smodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp\model_1_res_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ð
Umodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp^model_1_res_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ë
Dmodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV37model_1/res_block/cnn_block_5/conv2d_5/BiasAdd:output:0Jmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOp:value:0Lmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0[model_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0]model_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( ®
"model_1/res_block/cnn_block_5/ReluReluHmodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ñ
'model_1/res_block/max_pooling2d/MaxPoolMaxPool0model_1/res_block/cnn_block_5/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
Ï
>model_1/res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOpReadVariableOpGmodel_1_res_block_1_cnn_block_6_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
/model_1/res_block_1/cnn_block_6/conv2d_7/Conv2DConv2D0model_1/res_block/max_pooling2d/MaxPool:output:0Fmodel_1/res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
Å
?model_1/res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOpReadVariableOpHmodel_1_res_block_1_cnn_block_6_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ù
0model_1/res_block_1/cnn_block_6/conv2d_7/BiasAddBiasAdd8model_1/res_block_1/cnn_block_6/conv2d_7/Conv2D:output:0Gmodel_1/res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
Dmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOpMmodel_1_res_block_1_cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
Fmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOpOmodel_1_res_block_1_cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype0ñ
Umodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp^model_1_res_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0õ
Wmodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp`model_1_res_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ü
Fmodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV39model_1/res_block_1/cnn_block_6/conv2d_7/BiasAdd:output:0Lmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp:value:0Nmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0]model_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0_model_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ³
$model_1/res_block_1/cnn_block_6/ReluReluJmodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
>model_1/res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOpReadVariableOpGmodel_1_res_block_1_cnn_block_7_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
/model_1/res_block_1/cnn_block_7/conv2d_8/Conv2DConv2D2model_1/res_block_1/cnn_block_6/Relu:activations:0Fmodel_1/res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
Å
?model_1/res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOpReadVariableOpHmodel_1_res_block_1_cnn_block_7_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ù
0model_1/res_block_1/cnn_block_7/conv2d_8/BiasAddBiasAdd8model_1/res_block_1/cnn_block_7/conv2d_8/Conv2D:output:0Gmodel_1/res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
Dmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOpReadVariableOpMmodel_1_res_block_1_cnn_block_7_batch_normalization_7_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
Fmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1ReadVariableOpOmodel_1_res_block_1_cnn_block_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:*
dtype0ñ
Umodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp^model_1_res_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0õ
Wmodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp`model_1_res_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ü
Fmodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV39model_1/res_block_1/cnn_block_7/conv2d_8/BiasAdd:output:0Lmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp:value:0Nmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1:value:0]model_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0_model_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ³
$model_1/res_block_1/cnn_block_7/ReluReluJmodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
3model_1/res_block_1/conv2d_10/Conv2D/ReadVariableOpReadVariableOp<model_1_res_block_1_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
$model_1/res_block_1/conv2d_10/Conv2DConv2D0model_1/res_block/max_pooling2d/MaxPool:output:0;model_1/res_block_1/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¯
4model_1/res_block_1/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp=model_1_res_block_1_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_1/res_block_1/conv2d_10/BiasAddBiasAdd-model_1/res_block_1/conv2d_10/Conv2D:output:0<model_1/res_block_1/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
model_1/res_block_1/addAddV22model_1/res_block_1/cnn_block_7/Relu:activations:0.model_1/res_block_1/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
>model_1/res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOpReadVariableOpGmodel_1_res_block_1_cnn_block_8_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
/model_1/res_block_1/cnn_block_8/conv2d_9/Conv2DConv2Dmodel_1/res_block_1/add:z:0Fmodel_1/res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
Å
?model_1/res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOpReadVariableOpHmodel_1_res_block_1_cnn_block_8_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ù
0model_1/res_block_1/cnn_block_8/conv2d_9/BiasAddBiasAdd8model_1/res_block_1/cnn_block_8/conv2d_9/Conv2D:output:0Gmodel_1/res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
Dmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOpReadVariableOpMmodel_1_res_block_1_cnn_block_8_batch_normalization_8_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
Fmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1ReadVariableOpOmodel_1_res_block_1_cnn_block_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:*
dtype0ñ
Umodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp^model_1_res_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0õ
Wmodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp`model_1_res_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ü
Fmodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV39model_1/res_block_1/cnn_block_8/conv2d_9/BiasAdd:output:0Lmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp:value:0Nmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1:value:0]model_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0_model_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ³
$model_1/res_block_1/cnn_block_8/ReluReluJmodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
+model_1/res_block_1/max_pooling2d_1/MaxPoolMaxPool2model_1/res_block_1/cnn_block_8/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
h
model_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 1  ¯
model_1/flatten_1/ReshapeReshape4model_1/res_block_1/max_pooling2d_1/MaxPool:output:0 model_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
%model_1/dense_2/MatMul/ReadVariableOpReadVariableOp.model_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	b
*
dtype0¥
model_1/dense_2/MatMulMatMul"model_1/flatten_1/Reshape:output:0-model_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&model_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0¦
model_1/dense_2/BiasAddBiasAdd model_1/dense_2/MatMul:product:0.model_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
IdentityIdentity model_1/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
å
NoOpNoOp'^model_1/dense_2/BiasAdd/ReadVariableOp&^model_1/dense_2/MatMul/ReadVariableOpT^model_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpV^model_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1C^model_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOpE^model_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1>^model_1/res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp=^model_1/res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpT^model_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpV^model_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1C^model_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOpE^model_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1>^model_1/res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp=^model_1/res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpT^model_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpV^model_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1C^model_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOpE^model_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1>^model_1/res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp=^model_1/res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp2^model_1/res_block/conv2d_6/BiasAdd/ReadVariableOp1^model_1/res_block/conv2d_6/Conv2D/ReadVariableOpV^model_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpX^model_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1E^model_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOpG^model_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1@^model_1/res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp?^model_1/res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOpV^model_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpX^model_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1E^model_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOpG^model_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1@^model_1/res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp?^model_1/res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOpV^model_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpX^model_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1E^model_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOpG^model_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1@^model_1/res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp?^model_1/res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp5^model_1/res_block_1/conv2d_10/BiasAdd/ReadVariableOp4^model_1/res_block_1/conv2d_10/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesq
o:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&model_1/dense_2/BiasAdd/ReadVariableOp&model_1/dense_2/BiasAdd/ReadVariableOp2N
%model_1/dense_2/MatMul/ReadVariableOp%model_1/dense_2/MatMul/ReadVariableOp2ª
Smodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpSmodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2®
Umodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Umodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12
Bmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOpBmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOp2
Dmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1Dmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_12~
=model_1/res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp=model_1/res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp2|
<model_1/res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp<model_1/res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp2ª
Smodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpSmodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2®
Umodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Umodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12
Bmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOpBmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOp2
Dmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1Dmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_12~
=model_1/res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp=model_1/res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp2|
<model_1/res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp<model_1/res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp2ª
Smodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpSmodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2®
Umodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Umodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12
Bmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOpBmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOp2
Dmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1Dmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_12~
=model_1/res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp=model_1/res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp2|
<model_1/res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp<model_1/res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp2f
1model_1/res_block/conv2d_6/BiasAdd/ReadVariableOp1model_1/res_block/conv2d_6/BiasAdd/ReadVariableOp2d
0model_1/res_block/conv2d_6/Conv2D/ReadVariableOp0model_1/res_block/conv2d_6/Conv2D/ReadVariableOp2®
Umodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpUmodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2²
Wmodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Wmodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12
Dmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOpDmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp2
Fmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1Fmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_12
?model_1/res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp?model_1/res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp2
>model_1/res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp>model_1/res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp2®
Umodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpUmodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2²
Wmodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Wmodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12
Dmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOpDmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp2
Fmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1Fmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_12
?model_1/res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp?model_1/res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp2
>model_1/res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp>model_1/res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp2®
Umodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpUmodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2²
Wmodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Wmodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12
Dmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOpDmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp2
Fmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1Fmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_12
?model_1/res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp?model_1/res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp2
>model_1/res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp>model_1/res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp2l
4model_1/res_block_1/conv2d_10/BiasAdd/ReadVariableOp4model_1/res_block_1/conv2d_10/BiasAdd/ReadVariableOp2j
3model_1/res_block_1/conv2d_10/Conv2D/ReadVariableOp3model_1/res_block_1/conv2d_10/Conv2D/ReadVariableOp:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
	
Ô
5__inference_batch_normalization_7_layer_call_fn_10141

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
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
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7423
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
	
Ð
5__inference_batch_normalization_5_layer_call_fn_10017

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
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
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_7283
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
µ
H
,__inference_max_pooling2d_layer_call_fn_9852

inputs
identityÚ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7303
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¾
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9929

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
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
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
Ç
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_7757

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 1  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿbY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ó
A__inference_dense_2_layer_call_and_return_conditional_losses_7769

inputs1
matmul_readvariableop_resource:	b
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	b
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿb: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 
_user_specified_nameinputs
Û

P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10097

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
¹
J
.__inference_max_pooling2d_1_layer_call_fn_9862

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7507
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ô
5__inference_batch_normalization_6_layer_call_fn_10066

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
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
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7328
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
	
Ï
4__inference_batch_normalization_3_layer_call_fn_9893

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
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
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_7155
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
Ê


&__inference_model_1_layer_call_fn_7863
input_1!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: $

unknown_13: @

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@%

unknown_19:@

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:	

unknown_24:	&

unknown_25:

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	%

unknown_31:@

unknown_32:	&

unknown_33:

unknown_34:	

unknown_35:	

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:	b


unknown_40:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_7776o
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
_construction_contextkEagerRuntime*
_input_shapesq
o:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ð$
Ø
A__inference_model_1_layer_call_and_return_conditional_losses_8662
input_1(
res_block_8573: 
res_block_8575: 
res_block_8577: 
res_block_8579: 
res_block_8581: 
res_block_8583: (
res_block_8585:  
res_block_8587: 
res_block_8589: 
res_block_8591: 
res_block_8593: 
res_block_8595: (
res_block_8597: 
res_block_8599: (
res_block_8601: @
res_block_8603:@
res_block_8605:@
res_block_8607:@
res_block_8609:@
res_block_8611:@+
res_block_1_8614:@
res_block_1_8616:	
res_block_1_8618:	
res_block_1_8620:	
res_block_1_8622:	
res_block_1_8624:	,
res_block_1_8626:
res_block_1_8628:	
res_block_1_8630:	
res_block_1_8632:	
res_block_1_8634:	
res_block_1_8636:	+
res_block_1_8638:@
res_block_1_8640:	,
res_block_1_8642:
res_block_1_8644:	
res_block_1_8646:	
res_block_1_8648:	
res_block_1_8650:	
res_block_1_8652:	
dense_2_8656:	b

dense_2_8658:

identity¢dense_2/StatefulPartitionedCall¢!res_block/StatefulPartitionedCall¢#res_block_1/StatefulPartitionedCallÀ
!res_block/StatefulPartitionedCallStatefulPartitionedCallinput_1res_block_8573res_block_8575res_block_8577res_block_8579res_block_8581res_block_8583res_block_8585res_block_8587res_block_8589res_block_8591res_block_8593res_block_8595res_block_8597res_block_8599res_block_8601res_block_8603res_block_8605res_block_8607res_block_8609res_block_8611* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_res_block_layer_call_and_return_conditional_losses_7592
#res_block_1/StatefulPartitionedCallStatefulPartitionedCall*res_block/StatefulPartitionedCall:output:0res_block_1_8614res_block_1_8616res_block_1_8618res_block_1_8620res_block_1_8622res_block_1_8624res_block_1_8626res_block_1_8628res_block_1_8630res_block_1_8632res_block_1_8634res_block_1_8636res_block_1_8638res_block_1_8640res_block_1_8642res_block_1_8644res_block_1_8646res_block_1_8648res_block_1_8650res_block_1_8652* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_res_block_1_layer_call_and_return_conditional_losses_7709ä
flatten_1/PartitionedCallPartitionedCall,res_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_7757
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_8656dense_2_8658*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_7769w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
²
NoOpNoOp ^dense_2/StatefulPartitionedCall"^res_block/StatefulPartitionedCall$^res_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesq
o:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!res_block/StatefulPartitionedCall!res_block/StatefulPartitionedCall2J
#res_block_1/StatefulPartitionedCall#res_block_1/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Í$
×
A__inference_model_1_layer_call_and_return_conditional_losses_7776

inputs(
res_block_7593: 
res_block_7595: 
res_block_7597: 
res_block_7599: 
res_block_7601: 
res_block_7603: (
res_block_7605:  
res_block_7607: 
res_block_7609: 
res_block_7611: 
res_block_7613: 
res_block_7615: (
res_block_7617: 
res_block_7619: (
res_block_7621: @
res_block_7623:@
res_block_7625:@
res_block_7627:@
res_block_7629:@
res_block_7631:@+
res_block_1_7710:@
res_block_1_7712:	
res_block_1_7714:	
res_block_1_7716:	
res_block_1_7718:	
res_block_1_7720:	,
res_block_1_7722:
res_block_1_7724:	
res_block_1_7726:	
res_block_1_7728:	
res_block_1_7730:	
res_block_1_7732:	+
res_block_1_7734:@
res_block_1_7736:	,
res_block_1_7738:
res_block_1_7740:	
res_block_1_7742:	
res_block_1_7744:	
res_block_1_7746:	
res_block_1_7748:	
dense_2_7770:	b

dense_2_7772:

identity¢dense_2/StatefulPartitionedCall¢!res_block/StatefulPartitionedCall¢#res_block_1/StatefulPartitionedCall¿
!res_block/StatefulPartitionedCallStatefulPartitionedCallinputsres_block_7593res_block_7595res_block_7597res_block_7599res_block_7601res_block_7603res_block_7605res_block_7607res_block_7609res_block_7611res_block_7613res_block_7615res_block_7617res_block_7619res_block_7621res_block_7623res_block_7625res_block_7627res_block_7629res_block_7631* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_res_block_layer_call_and_return_conditional_losses_7592
#res_block_1/StatefulPartitionedCallStatefulPartitionedCall*res_block/StatefulPartitionedCall:output:0res_block_1_7710res_block_1_7712res_block_1_7714res_block_1_7716res_block_1_7718res_block_1_7720res_block_1_7722res_block_1_7724res_block_1_7726res_block_1_7728res_block_1_7730res_block_1_7732res_block_1_7734res_block_1_7736res_block_1_7738res_block_1_7740res_block_1_7742res_block_1_7744res_block_1_7746res_block_1_7748* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_res_block_1_layer_call_and_return_conditional_losses_7709ä
flatten_1/PartitionedCallPartitionedCall,res_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_7757
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_7770dense_2_7772*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_7769w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
²
NoOpNoOp ^dense_2/StatefulPartitionedCall"^res_block/StatefulPartitionedCall$^res_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesq
o:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!res_block/StatefulPartitionedCall!res_block/StatefulPartitionedCall2J
#res_block_1/StatefulPartitionedCall#res_block_1/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ï
4__inference_batch_normalization_4_layer_call_fn_9942

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
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
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_7188
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
ý
2
A__inference_model_1_layer_call_and_return_conditional_losses_9183

inputsW
=res_block_cnn_block_3_conv2d_3_conv2d_readvariableop_resource: L
>res_block_cnn_block_3_conv2d_3_biasadd_readvariableop_resource: Q
Cres_block_cnn_block_3_batch_normalization_3_readvariableop_resource: S
Eres_block_cnn_block_3_batch_normalization_3_readvariableop_1_resource: b
Tres_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource: d
Vres_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: W
=res_block_cnn_block_4_conv2d_4_conv2d_readvariableop_resource:  L
>res_block_cnn_block_4_conv2d_4_biasadd_readvariableop_resource: Q
Cres_block_cnn_block_4_batch_normalization_4_readvariableop_resource: S
Eres_block_cnn_block_4_batch_normalization_4_readvariableop_1_resource: b
Tres_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource: d
Vres_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: K
1res_block_conv2d_6_conv2d_readvariableop_resource: @
2res_block_conv2d_6_biasadd_readvariableop_resource: W
=res_block_cnn_block_5_conv2d_5_conv2d_readvariableop_resource: @L
>res_block_cnn_block_5_conv2d_5_biasadd_readvariableop_resource:@Q
Cres_block_cnn_block_5_batch_normalization_5_readvariableop_resource:@S
Eres_block_cnn_block_5_batch_normalization_5_readvariableop_1_resource:@b
Tres_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@d
Vres_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@Z
?res_block_1_cnn_block_6_conv2d_7_conv2d_readvariableop_resource:@O
@res_block_1_cnn_block_6_conv2d_7_biasadd_readvariableop_resource:	T
Eres_block_1_cnn_block_6_batch_normalization_6_readvariableop_resource:	V
Gres_block_1_cnn_block_6_batch_normalization_6_readvariableop_1_resource:	e
Vres_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	g
Xres_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	[
?res_block_1_cnn_block_7_conv2d_8_conv2d_readvariableop_resource:O
@res_block_1_cnn_block_7_conv2d_8_biasadd_readvariableop_resource:	T
Eres_block_1_cnn_block_7_batch_normalization_7_readvariableop_resource:	V
Gres_block_1_cnn_block_7_batch_normalization_7_readvariableop_1_resource:	e
Vres_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	g
Xres_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	O
4res_block_1_conv2d_10_conv2d_readvariableop_resource:@D
5res_block_1_conv2d_10_biasadd_readvariableop_resource:	[
?res_block_1_cnn_block_8_conv2d_9_conv2d_readvariableop_resource:O
@res_block_1_cnn_block_8_conv2d_9_biasadd_readvariableop_resource:	T
Eres_block_1_cnn_block_8_batch_normalization_8_readvariableop_resource:	V
Gres_block_1_cnn_block_8_batch_normalization_8_readvariableop_1_resource:	e
Vres_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	g
Xres_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	9
&dense_2_matmul_readvariableop_resource:	b
5
'dense_2_biasadd_readvariableop_resource:

identity¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢Kres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢Mres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢:res_block/cnn_block_3/batch_normalization_3/ReadVariableOp¢<res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1¢5res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp¢4res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp¢Kres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp¢Mres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1¢:res_block/cnn_block_4/batch_normalization_4/ReadVariableOp¢<res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1¢5res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp¢4res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp¢Kres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp¢Mres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1¢:res_block/cnn_block_5/batch_normalization_5/ReadVariableOp¢<res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1¢5res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp¢4res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp¢)res_block/conv2d_6/BiasAdd/ReadVariableOp¢(res_block/conv2d_6/Conv2D/ReadVariableOp¢Mres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp¢Ores_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1¢<res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp¢>res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1¢7res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp¢6res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp¢Mres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp¢Ores_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1¢<res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp¢>res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1¢7res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp¢6res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp¢Mres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢Ores_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢<res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp¢>res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1¢7res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp¢6res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp¢,res_block_1/conv2d_10/BiasAdd/ReadVariableOp¢+res_block_1/conv2d_10/Conv2D/ReadVariableOpº
4res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp=res_block_cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0×
%res_block/cnn_block_3/conv2d_3/Conv2DConv2Dinputs<res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
°
5res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp>res_block_cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ú
&res_block/cnn_block_3/conv2d_3/BiasAddBiasAdd.res_block/cnn_block_3/conv2d_3/Conv2D:output:0=res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ º
:res_block/cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOpCres_block_cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype0¾
<res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOpEres_block_cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ü
Kres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpTres_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0à
Mres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVres_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0»
<res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3/res_block/cnn_block_3/conv2d_3/BiasAdd:output:0Bres_block/cnn_block_3/batch_normalization_3/ReadVariableOp:value:0Dres_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0Sres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Ures_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 
res_block/cnn_block_3/ReluRelu@res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ º
4res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp=res_block_cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0ù
%res_block/cnn_block_4/conv2d_4/Conv2DConv2D(res_block/cnn_block_3/Relu:activations:0<res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
°
5res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp>res_block_cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ú
&res_block/cnn_block_4/conv2d_4/BiasAddBiasAdd.res_block/cnn_block_4/conv2d_4/Conv2D:output:0=res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ º
:res_block/cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOpCres_block_cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype0¾
<res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOpEres_block_cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype0Ü
Kres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpTres_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0à
Mres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVres_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0»
<res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3/res_block/cnn_block_4/conv2d_4/BiasAdd:output:0Bres_block/cnn_block_4/batch_normalization_4/ReadVariableOp:value:0Dres_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0Sres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Ures_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 
res_block/cnn_block_4/ReluRelu@res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¢
(res_block/conv2d_6/Conv2D/ReadVariableOpReadVariableOp1res_block_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0¿
res_block/conv2d_6/Conv2DConv2Dinputs0res_block/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

)res_block/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp2res_block_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¶
res_block/conv2d_6/BiasAddBiasAdd"res_block/conv2d_6/Conv2D:output:01res_block/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
res_block/addAddV2(res_block/cnn_block_4/Relu:activations:0#res_block/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ º
4res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp=res_block_cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0â
%res_block/cnn_block_5/conv2d_5/Conv2DConv2Dres_block/add:z:0<res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
°
5res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp>res_block_cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ú
&res_block/cnn_block_5/conv2d_5/BiasAddBiasAdd.res_block/cnn_block_5/conv2d_5/Conv2D:output:0=res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
:res_block/cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOpCres_block_cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0¾
<res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOpEres_block_cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ü
Kres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpTres_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0à
Mres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVres_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
<res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3/res_block/cnn_block_5/conv2d_5/BiasAdd:output:0Bres_block/cnn_block_5/batch_normalization_5/ReadVariableOp:value:0Dres_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0Sres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Ures_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 
res_block/cnn_block_5/ReluRelu@res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Á
res_block/max_pooling2d/MaxPoolMaxPool(res_block/cnn_block_5/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
¿
6res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOpReadVariableOp?res_block_1_cnn_block_6_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0þ
'res_block_1/cnn_block_6/conv2d_7/Conv2DConv2D(res_block/max_pooling2d/MaxPool:output:0>res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
µ
7res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp@res_block_1_cnn_block_6_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0á
(res_block_1/cnn_block_6/conv2d_7/BiasAddBiasAdd0res_block_1/cnn_block_6/conv2d_7/Conv2D:output:0?res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
<res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOpEres_block_1_cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
>res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOpGres_block_1_cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype0á
Mres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpVres_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0å
Ores_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpXres_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ì
>res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV31res_block_1/cnn_block_6/conv2d_7/BiasAdd:output:0Dres_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp:value:0Fres_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0Ures_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Wres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( £
res_block_1/cnn_block_6/ReluReluBres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
6res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOpReadVariableOp?res_block_1_cnn_block_7_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
'res_block_1/cnn_block_7/conv2d_8/Conv2DConv2D*res_block_1/cnn_block_6/Relu:activations:0>res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
µ
7res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp@res_block_1_cnn_block_7_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0á
(res_block_1/cnn_block_7/conv2d_8/BiasAddBiasAdd0res_block_1/cnn_block_7/conv2d_8/Conv2D:output:0?res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
<res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOpReadVariableOpEres_block_1_cnn_block_7_batch_normalization_7_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
>res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1ReadVariableOpGres_block_1_cnn_block_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:*
dtype0á
Mres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpVres_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0å
Ores_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpXres_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ì
>res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV31res_block_1/cnn_block_7/conv2d_8/BiasAdd:output:0Dres_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp:value:0Fres_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1:value:0Ures_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Wres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( £
res_block_1/cnn_block_7/ReluReluBres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
+res_block_1/conv2d_10/Conv2D/ReadVariableOpReadVariableOp4res_block_1_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0è
res_block_1/conv2d_10/Conv2DConv2D(res_block/max_pooling2d/MaxPool:output:03res_block_1/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

,res_block_1/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp5res_block_1_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0À
res_block_1/conv2d_10/BiasAddBiasAdd%res_block_1/conv2d_10/Conv2D:output:04res_block_1/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
res_block_1/addAddV2*res_block_1/cnn_block_7/Relu:activations:0&res_block_1/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
6res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOpReadVariableOp?res_block_1_cnn_block_8_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0é
'res_block_1/cnn_block_8/conv2d_9/Conv2DConv2Dres_block_1/add:z:0>res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
µ
7res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp@res_block_1_cnn_block_8_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0á
(res_block_1/cnn_block_8/conv2d_9/BiasAddBiasAdd0res_block_1/cnn_block_8/conv2d_9/Conv2D:output:0?res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
<res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOpReadVariableOpEres_block_1_cnn_block_8_batch_normalization_8_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
>res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1ReadVariableOpGres_block_1_cnn_block_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:*
dtype0á
Mres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpVres_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0å
Ores_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpXres_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ì
>res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV31res_block_1/cnn_block_8/conv2d_9/BiasAdd:output:0Dres_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp:value:0Fres_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1:value:0Ures_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Wres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( £
res_block_1/cnn_block_8/ReluReluBres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
#res_block_1/max_pooling2d_1/MaxPoolMaxPool*res_block_1/cnn_block_8/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 1  
flatten_1/ReshapeReshape,res_block_1/max_pooling2d_1/MaxPool:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	b
*
dtype0
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOpL^res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpN^res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1;^res_block/cnn_block_3/batch_normalization_3/ReadVariableOp=^res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_16^res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp5^res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpL^res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpN^res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1;^res_block/cnn_block_4/batch_normalization_4/ReadVariableOp=^res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_16^res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp5^res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpL^res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpN^res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1;^res_block/cnn_block_5/batch_normalization_5/ReadVariableOp=^res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_16^res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp5^res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp*^res_block/conv2d_6/BiasAdd/ReadVariableOp)^res_block/conv2d_6/Conv2D/ReadVariableOpN^res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpP^res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1=^res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp?^res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_18^res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp7^res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOpN^res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpP^res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1=^res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp?^res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_18^res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp7^res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOpN^res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpP^res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1=^res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp?^res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_18^res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp7^res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp-^res_block_1/conv2d_10/BiasAdd/ReadVariableOp,^res_block_1/conv2d_10/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesq
o:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2
Kres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpKres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2
Mres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Mres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12x
:res_block/cnn_block_3/batch_normalization_3/ReadVariableOp:res_block/cnn_block_3/batch_normalization_3/ReadVariableOp2|
<res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1<res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_12n
5res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp5res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp2l
4res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp4res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp2
Kres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpKres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2
Mres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Mres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12x
:res_block/cnn_block_4/batch_normalization_4/ReadVariableOp:res_block/cnn_block_4/batch_normalization_4/ReadVariableOp2|
<res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1<res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_12n
5res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp5res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp2l
4res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp4res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp2
Kres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpKres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2
Mres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Mres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12x
:res_block/cnn_block_5/batch_normalization_5/ReadVariableOp:res_block/cnn_block_5/batch_normalization_5/ReadVariableOp2|
<res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1<res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_12n
5res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp5res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp2l
4res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp4res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp2V
)res_block/conv2d_6/BiasAdd/ReadVariableOp)res_block/conv2d_6/BiasAdd/ReadVariableOp2T
(res_block/conv2d_6/Conv2D/ReadVariableOp(res_block/conv2d_6/Conv2D/ReadVariableOp2
Mres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpMres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2¢
Ores_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ores_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12|
<res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp<res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp2
>res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1>res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_12r
7res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp7res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp2p
6res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp6res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp2
Mres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpMres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2¢
Ores_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ores_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12|
<res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp<res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp2
>res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1>res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_12r
7res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp7res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp2p
6res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp6res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp2
Mres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpMres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2¢
Ores_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ores_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12|
<res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp<res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp2
>res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1>res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_12r
7res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp7res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp2p
6res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp6res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp2\
,res_block_1/conv2d_10/BiasAdd/ReadVariableOp,res_block_1/conv2d_10/BiasAdd/ReadVariableOp2Z
+res_block_1/conv2d_10/Conv2D/ReadVariableOp+res_block_1/conv2d_10/Conv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë

P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10035

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

Â
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7423

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
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
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
Í$
×
A__inference_model_1_layer_call_and_return_conditional_losses_8394

inputs(
res_block_8305: 
res_block_8307: 
res_block_8309: 
res_block_8311: 
res_block_8313: 
res_block_8315: (
res_block_8317:  
res_block_8319: 
res_block_8321: 
res_block_8323: 
res_block_8325: 
res_block_8327: (
res_block_8329: 
res_block_8331: (
res_block_8333: @
res_block_8335:@
res_block_8337:@
res_block_8339:@
res_block_8341:@
res_block_8343:@+
res_block_1_8346:@
res_block_1_8348:	
res_block_1_8350:	
res_block_1_8352:	
res_block_1_8354:	
res_block_1_8356:	,
res_block_1_8358:
res_block_1_8360:	
res_block_1_8362:	
res_block_1_8364:	
res_block_1_8366:	
res_block_1_8368:	+
res_block_1_8370:@
res_block_1_8372:	,
res_block_1_8374:
res_block_1_8376:	
res_block_1_8378:	
res_block_1_8380:	
res_block_1_8382:	
res_block_1_8384:	
dense_2_8388:	b

dense_2_8390:

identity¢dense_2/StatefulPartitionedCall¢!res_block/StatefulPartitionedCall¢#res_block_1/StatefulPartitionedCall¿
!res_block/StatefulPartitionedCallStatefulPartitionedCallinputsres_block_8305res_block_8307res_block_8309res_block_8311res_block_8313res_block_8315res_block_8317res_block_8319res_block_8321res_block_8323res_block_8325res_block_8327res_block_8329res_block_8331res_block_8333res_block_8335res_block_8337res_block_8339res_block_8341res_block_8343* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_res_block_layer_call_and_return_conditional_losses_7592
#res_block_1/StatefulPartitionedCallStatefulPartitionedCall*res_block/StatefulPartitionedCall:output:0res_block_1_8346res_block_1_8348res_block_1_8350res_block_1_8352res_block_1_8354res_block_1_8356res_block_1_8358res_block_1_8360res_block_1_8362res_block_1_8364res_block_1_8366res_block_1_8368res_block_1_8370res_block_1_8372res_block_1_8374res_block_1_8376res_block_1_8378res_block_1_8380res_block_1_8382res_block_1_8384* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_res_block_1_layer_call_and_return_conditional_losses_7709ä
flatten_1/PartitionedCallPartitionedCall,res_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_7757
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_8388dense_2_8390*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_7769w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
²
NoOpNoOp ^dense_2/StatefulPartitionedCall"^res_block/StatefulPartitionedCall$^res_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesq
o:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!res_block/StatefulPartitionedCall!res_block/StatefulPartitionedCall2J
#res_block_1/StatefulPartitionedCall#res_block_1/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ô
5__inference_batch_normalization_7_layer_call_fn_10128

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
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
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7392
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
´
D
(__inference_flatten_1_layer_call_fn_9822

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_7757a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7507

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
ø
E__inference_res_block_1_layer_call_and_return_conditional_losses_8002
input_tensorN
3cnn_block_6_conv2d_7_conv2d_readvariableop_resource:@C
4cnn_block_6_conv2d_7_biasadd_readvariableop_resource:	H
9cnn_block_6_batch_normalization_6_readvariableop_resource:	J
;cnn_block_6_batch_normalization_6_readvariableop_1_resource:	Y
Jcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	[
Lcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	O
3cnn_block_7_conv2d_8_conv2d_readvariableop_resource:C
4cnn_block_7_conv2d_8_biasadd_readvariableop_resource:	H
9cnn_block_7_batch_normalization_7_readvariableop_resource:	J
;cnn_block_7_batch_normalization_7_readvariableop_1_resource:	Y
Jcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	[
Lcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	C
(conv2d_10_conv2d_readvariableop_resource:@8
)conv2d_10_biasadd_readvariableop_resource:	O
3cnn_block_8_conv2d_9_conv2d_readvariableop_resource:C
4cnn_block_8_conv2d_9_biasadd_readvariableop_resource:	H
9cnn_block_8_batch_normalization_8_readvariableop_resource:	J
;cnn_block_8_batch_normalization_8_readvariableop_1_resource:	Y
Jcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	[
Lcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	
identity¢0cnn_block_6/batch_normalization_6/AssignNewValue¢2cnn_block_6/batch_normalization_6/AssignNewValue_1¢Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_6/batch_normalization_6/ReadVariableOp¢2cnn_block_6/batch_normalization_6/ReadVariableOp_1¢+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp¢*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp¢0cnn_block_7/batch_normalization_7/AssignNewValue¢2cnn_block_7/batch_normalization_7/AssignNewValue_1¢Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_7/batch_normalization_7/ReadVariableOp¢2cnn_block_7/batch_normalization_7/ReadVariableOp_1¢+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp¢*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp¢0cnn_block_8/batch_normalization_8/AssignNewValue¢2cnn_block_8/batch_normalization_8/AssignNewValue_1¢Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_8/batch_normalization_8/ReadVariableOp¢2cnn_block_8/batch_normalization_8/ReadVariableOp_1¢+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp¢*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp¢ conv2d_10/BiasAdd/ReadVariableOp¢conv2d_10/Conv2D/ReadVariableOp§
*cnn_block_6/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3cnn_block_6_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ê
cnn_block_6/conv2d_7/Conv2DConv2Dinput_tensor2cnn_block_6/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_6_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
cnn_block_6/conv2d_7/BiasAddBiasAdd$cnn_block_6/conv2d_7/Conv2D:output:03cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOp9cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype0«
2cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp;cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype0É
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Í
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
2cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%cnn_block_6/conv2d_7/BiasAdd:output:08cnn_block_6/batch_normalization_6/ReadVariableOp:value:0:cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0Icnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Î
0cnn_block_6/batch_normalization_6/AssignNewValueAssignVariableOpJcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource?cnn_block_6/batch_normalization_6/FusedBatchNormV3:batch_mean:0B^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ø
2cnn_block_6/batch_normalization_6/AssignNewValue_1AssignVariableOpLcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_6/batch_normalization_6/FusedBatchNormV3:batch_variance:0D^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
cnn_block_6/ReluRelu6cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
*cnn_block_7/conv2d_8/Conv2D/ReadVariableOpReadVariableOp3cnn_block_7_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
cnn_block_7/conv2d_8/Conv2DConv2Dcnn_block_6/Relu:activations:02cnn_block_7/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_7_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
cnn_block_7/conv2d_8/BiasAddBiasAdd$cnn_block_7/conv2d_8/Conv2D:output:03cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0cnn_block_7/batch_normalization_7/ReadVariableOpReadVariableOp9cnn_block_7_batch_normalization_7_readvariableop_resource*
_output_shapes	
:*
dtype0«
2cnn_block_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp;cnn_block_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:*
dtype0É
Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Í
Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
2cnn_block_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%cnn_block_7/conv2d_8/BiasAdd:output:08cnn_block_7/batch_normalization_7/ReadVariableOp:value:0:cnn_block_7/batch_normalization_7/ReadVariableOp_1:value:0Icnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Î
0cnn_block_7/batch_normalization_7/AssignNewValueAssignVariableOpJcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource?cnn_block_7/batch_normalization_7/FusedBatchNormV3:batch_mean:0B^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ø
2cnn_block_7/batch_normalization_7/AssignNewValue_1AssignVariableOpLcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_7/batch_normalization_7/FusedBatchNormV3:batch_variance:0D^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
cnn_block_7/ReluRelu6cnn_block_7/batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0´
conv2d_10/Conv2DConv2Dinput_tensor'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
addAddV2cnn_block_7/Relu:activations:0conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
*cnn_block_8/conv2d_9/Conv2D/ReadVariableOpReadVariableOp3cnn_block_8_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Å
cnn_block_8/conv2d_9/Conv2DConv2Dadd:z:02cnn_block_8/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_8_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
cnn_block_8/conv2d_9/BiasAddBiasAdd$cnn_block_8/conv2d_9/Conv2D:output:03cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0cnn_block_8/batch_normalization_8/ReadVariableOpReadVariableOp9cnn_block_8_batch_normalization_8_readvariableop_resource*
_output_shapes	
:*
dtype0«
2cnn_block_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp;cnn_block_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:*
dtype0É
Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Í
Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
2cnn_block_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3%cnn_block_8/conv2d_9/BiasAdd:output:08cnn_block_8/batch_normalization_8/ReadVariableOp:value:0:cnn_block_8/batch_normalization_8/ReadVariableOp_1:value:0Icnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Î
0cnn_block_8/batch_normalization_8/AssignNewValueAssignVariableOpJcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource?cnn_block_8/batch_normalization_8/FusedBatchNormV3:batch_mean:0B^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ø
2cnn_block_8/batch_normalization_8/AssignNewValue_1AssignVariableOpLcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_8/batch_normalization_8/FusedBatchNormV3:batch_variance:0D^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
cnn_block_8/ReluRelu6cnn_block_8/batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
max_pooling2d_1/MaxPoolMaxPoolcnn_block_8/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
x
IdentityIdentity max_pooling2d_1/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
NoOpNoOp1^cnn_block_6/batch_normalization_6/AssignNewValue3^cnn_block_6/batch_normalization_6/AssignNewValue_1B^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpD^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_11^cnn_block_6/batch_normalization_6/ReadVariableOp3^cnn_block_6/batch_normalization_6/ReadVariableOp_1,^cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp+^cnn_block_6/conv2d_7/Conv2D/ReadVariableOp1^cnn_block_7/batch_normalization_7/AssignNewValue3^cnn_block_7/batch_normalization_7/AssignNewValue_1B^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpD^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_11^cnn_block_7/batch_normalization_7/ReadVariableOp3^cnn_block_7/batch_normalization_7/ReadVariableOp_1,^cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp+^cnn_block_7/conv2d_8/Conv2D/ReadVariableOp1^cnn_block_8/batch_normalization_8/AssignNewValue3^cnn_block_8/batch_normalization_8/AssignNewValue_1B^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpD^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_11^cnn_block_8/batch_normalization_8/ReadVariableOp3^cnn_block_8/batch_normalization_8/ReadVariableOp_1,^cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp+^cnn_block_8/conv2d_9/Conv2D/ReadVariableOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ@: : : : : : : : : : : : : : : : : : : : 2d
0cnn_block_6/batch_normalization_6/AssignNewValue0cnn_block_6/batch_normalization_6/AssignNewValue2h
2cnn_block_6/batch_normalization_6/AssignNewValue_12cnn_block_6/batch_normalization_6/AssignNewValue_12
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpAcnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_6/batch_normalization_6/ReadVariableOp0cnn_block_6/batch_normalization_6/ReadVariableOp2h
2cnn_block_6/batch_normalization_6/ReadVariableOp_12cnn_block_6/batch_normalization_6/ReadVariableOp_12Z
+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp2X
*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp2d
0cnn_block_7/batch_normalization_7/AssignNewValue0cnn_block_7/batch_normalization_7/AssignNewValue2h
2cnn_block_7/batch_normalization_7/AssignNewValue_12cnn_block_7/batch_normalization_7/AssignNewValue_12
Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpAcnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_7/batch_normalization_7/ReadVariableOp0cnn_block_7/batch_normalization_7/ReadVariableOp2h
2cnn_block_7/batch_normalization_7/ReadVariableOp_12cnn_block_7/batch_normalization_7/ReadVariableOp_12Z
+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp2X
*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp2d
0cnn_block_8/batch_normalization_8/AssignNewValue0cnn_block_8/batch_normalization_8/AssignNewValue2h
2cnn_block_8/batch_normalization_8/AssignNewValue_12cnn_block_8/batch_normalization_8/AssignNewValue_12
Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpAcnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_8/batch_normalization_8/ReadVariableOp0cnn_block_8/batch_normalization_8/ReadVariableOp2h
2cnn_block_8/batch_normalization_8/ReadVariableOp_12cnn_block_8/batch_normalization_8/ReadVariableOp_12Z
+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp2X
*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
&
_user_specified_nameinput_tensor

¾
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_7283

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
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
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
Ê

O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9973

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
Ê

O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9911

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

Ã
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10177

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
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
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
ìk
À
E__inference_res_block_1_layer_call_and_return_conditional_losses_9742
input_tensorN
3cnn_block_6_conv2d_7_conv2d_readvariableop_resource:@C
4cnn_block_6_conv2d_7_biasadd_readvariableop_resource:	H
9cnn_block_6_batch_normalization_6_readvariableop_resource:	J
;cnn_block_6_batch_normalization_6_readvariableop_1_resource:	Y
Jcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	[
Lcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	O
3cnn_block_7_conv2d_8_conv2d_readvariableop_resource:C
4cnn_block_7_conv2d_8_biasadd_readvariableop_resource:	H
9cnn_block_7_batch_normalization_7_readvariableop_resource:	J
;cnn_block_7_batch_normalization_7_readvariableop_1_resource:	Y
Jcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	[
Lcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	C
(conv2d_10_conv2d_readvariableop_resource:@8
)conv2d_10_biasadd_readvariableop_resource:	O
3cnn_block_8_conv2d_9_conv2d_readvariableop_resource:C
4cnn_block_8_conv2d_9_biasadd_readvariableop_resource:	H
9cnn_block_8_batch_normalization_8_readvariableop_resource:	J
;cnn_block_8_batch_normalization_8_readvariableop_1_resource:	Y
Jcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	[
Lcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	
identity¢Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_6/batch_normalization_6/ReadVariableOp¢2cnn_block_6/batch_normalization_6/ReadVariableOp_1¢+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp¢*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp¢Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_7/batch_normalization_7/ReadVariableOp¢2cnn_block_7/batch_normalization_7/ReadVariableOp_1¢+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp¢*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp¢Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_8/batch_normalization_8/ReadVariableOp¢2cnn_block_8/batch_normalization_8/ReadVariableOp_1¢+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp¢*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp¢ conv2d_10/BiasAdd/ReadVariableOp¢conv2d_10/Conv2D/ReadVariableOp§
*cnn_block_6/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3cnn_block_6_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ê
cnn_block_6/conv2d_7/Conv2DConv2Dinput_tensor2cnn_block_6/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_6_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
cnn_block_6/conv2d_7/BiasAddBiasAdd$cnn_block_6/conv2d_7/Conv2D:output:03cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOp9cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype0«
2cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp;cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype0É
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Í
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
2cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%cnn_block_6/conv2d_7/BiasAdd:output:08cnn_block_6/batch_normalization_6/ReadVariableOp:value:0:cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0Icnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 
cnn_block_6/ReluRelu6cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
*cnn_block_7/conv2d_8/Conv2D/ReadVariableOpReadVariableOp3cnn_block_7_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
cnn_block_7/conv2d_8/Conv2DConv2Dcnn_block_6/Relu:activations:02cnn_block_7/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_7_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
cnn_block_7/conv2d_8/BiasAddBiasAdd$cnn_block_7/conv2d_8/Conv2D:output:03cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0cnn_block_7/batch_normalization_7/ReadVariableOpReadVariableOp9cnn_block_7_batch_normalization_7_readvariableop_resource*
_output_shapes	
:*
dtype0«
2cnn_block_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp;cnn_block_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:*
dtype0É
Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Í
Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
2cnn_block_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%cnn_block_7/conv2d_8/BiasAdd:output:08cnn_block_7/batch_normalization_7/ReadVariableOp:value:0:cnn_block_7/batch_normalization_7/ReadVariableOp_1:value:0Icnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 
cnn_block_7/ReluRelu6cnn_block_7/batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0´
conv2d_10/Conv2DConv2Dinput_tensor'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
addAddV2cnn_block_7/Relu:activations:0conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
*cnn_block_8/conv2d_9/Conv2D/ReadVariableOpReadVariableOp3cnn_block_8_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Å
cnn_block_8/conv2d_9/Conv2DConv2Dadd:z:02cnn_block_8/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_8_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
cnn_block_8/conv2d_9/BiasAddBiasAdd$cnn_block_8/conv2d_9/Conv2D:output:03cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0cnn_block_8/batch_normalization_8/ReadVariableOpReadVariableOp9cnn_block_8_batch_normalization_8_readvariableop_resource*
_output_shapes	
:*
dtype0«
2cnn_block_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp;cnn_block_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:*
dtype0É
Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Í
Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
2cnn_block_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3%cnn_block_8/conv2d_9/BiasAdd:output:08cnn_block_8/batch_normalization_8/ReadVariableOp:value:0:cnn_block_8/batch_normalization_8/ReadVariableOp_1:value:0Icnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 
cnn_block_8/ReluRelu6cnn_block_8/batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
max_pooling2d_1/MaxPoolMaxPoolcnn_block_8/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
x
IdentityIdentity max_pooling2d_1/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
NoOpNoOpB^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpD^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_11^cnn_block_6/batch_normalization_6/ReadVariableOp3^cnn_block_6/batch_normalization_6/ReadVariableOp_1,^cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp+^cnn_block_6/conv2d_7/Conv2D/ReadVariableOpB^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpD^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_11^cnn_block_7/batch_normalization_7/ReadVariableOp3^cnn_block_7/batch_normalization_7/ReadVariableOp_1,^cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp+^cnn_block_7/conv2d_8/Conv2D/ReadVariableOpB^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpD^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_11^cnn_block_8/batch_normalization_8/ReadVariableOp3^cnn_block_8/batch_normalization_8/ReadVariableOp_1,^cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp+^cnn_block_8/conv2d_9/Conv2D/ReadVariableOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ@: : : : : : : : : : : : : : : : : : : : 2
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpAcnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_6/batch_normalization_6/ReadVariableOp0cnn_block_6/batch_normalization_6/ReadVariableOp2h
2cnn_block_6/batch_normalization_6/ReadVariableOp_12cnn_block_6/batch_normalization_6/ReadVariableOp_12Z
+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp2X
*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp2
Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpAcnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_7/batch_normalization_7/ReadVariableOp0cnn_block_7/batch_normalization_7/ReadVariableOp2h
2cnn_block_7/batch_normalization_7/ReadVariableOp_12cnn_block_7/batch_normalization_7/ReadVariableOp_12Z
+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp2X
*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp2
Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpAcnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_8/batch_normalization_8/ReadVariableOp0cnn_block_8/batch_normalization_8/ReadVariableOp2h
2cnn_block_8/batch_normalization_8/ReadVariableOp_12cnn_block_8/batch_normalization_8/ReadVariableOp_12Z
+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp2X
*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
&
_user_specified_nameinput_tensor
Ð$
Ø
A__inference_model_1_layer_call_and_return_conditional_losses_8754
input_1(
res_block_8665: 
res_block_8667: 
res_block_8669: 
res_block_8671: 
res_block_8673: 
res_block_8675: (
res_block_8677:  
res_block_8679: 
res_block_8681: 
res_block_8683: 
res_block_8685: 
res_block_8687: (
res_block_8689: 
res_block_8691: (
res_block_8693: @
res_block_8695:@
res_block_8697:@
res_block_8699:@
res_block_8701:@
res_block_8703:@+
res_block_1_8706:@
res_block_1_8708:	
res_block_1_8710:	
res_block_1_8712:	
res_block_1_8714:	
res_block_1_8716:	,
res_block_1_8718:
res_block_1_8720:	
res_block_1_8722:	
res_block_1_8724:	
res_block_1_8726:	
res_block_1_8728:	+
res_block_1_8730:@
res_block_1_8732:	,
res_block_1_8734:
res_block_1_8736:	
res_block_1_8738:	
res_block_1_8740:	
res_block_1_8742:	
res_block_1_8744:	
dense_2_8748:	b

dense_2_8750:

identity¢dense_2/StatefulPartitionedCall¢!res_block/StatefulPartitionedCall¢#res_block_1/StatefulPartitionedCallÀ
!res_block/StatefulPartitionedCallStatefulPartitionedCallinput_1res_block_8665res_block_8667res_block_8669res_block_8671res_block_8673res_block_8675res_block_8677res_block_8679res_block_8681res_block_8683res_block_8685res_block_8687res_block_8689res_block_8691res_block_8693res_block_8695res_block_8697res_block_8699res_block_8701res_block_8703* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_res_block_layer_call_and_return_conditional_losses_7592
#res_block_1/StatefulPartitionedCallStatefulPartitionedCall*res_block/StatefulPartitionedCall:output:0res_block_1_8706res_block_1_8708res_block_1_8710res_block_1_8712res_block_1_8714res_block_1_8716res_block_1_8718res_block_1_8720res_block_1_8722res_block_1_8724res_block_1_8726res_block_1_8728res_block_1_8730res_block_1_8732res_block_1_8734res_block_1_8736res_block_1_8738res_block_1_8740res_block_1_8742res_block_1_8744* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_res_block_1_layer_call_and_return_conditional_losses_7709ä
flatten_1/PartitionedCallPartitionedCall,res_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_7757
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_8748dense_2_8750*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_7769w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
²
NoOpNoOp ^dense_2/StatefulPartitionedCall"^res_block/StatefulPartitionedCall$^res_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesq
o:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!res_block/StatefulPartitionedCall!res_block/StatefulPartitionedCall2J
#res_block_1/StatefulPartitionedCall#res_block_1/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ú

O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7456

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ô
5__inference_batch_normalization_8_layer_call_fn_10190

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7456
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ô
5__inference_batch_normalization_8_layer_call_fn_10203

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7487
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_9857

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_9828

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 1  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿbY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Â
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7487

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¿
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10053

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
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
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
ìk
À
E__inference_res_block_1_layer_call_and_return_conditional_losses_7709
input_tensorN
3cnn_block_6_conv2d_7_conv2d_readvariableop_resource:@C
4cnn_block_6_conv2d_7_biasadd_readvariableop_resource:	H
9cnn_block_6_batch_normalization_6_readvariableop_resource:	J
;cnn_block_6_batch_normalization_6_readvariableop_1_resource:	Y
Jcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	[
Lcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	O
3cnn_block_7_conv2d_8_conv2d_readvariableop_resource:C
4cnn_block_7_conv2d_8_biasadd_readvariableop_resource:	H
9cnn_block_7_batch_normalization_7_readvariableop_resource:	J
;cnn_block_7_batch_normalization_7_readvariableop_1_resource:	Y
Jcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	[
Lcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	C
(conv2d_10_conv2d_readvariableop_resource:@8
)conv2d_10_biasadd_readvariableop_resource:	O
3cnn_block_8_conv2d_9_conv2d_readvariableop_resource:C
4cnn_block_8_conv2d_9_biasadd_readvariableop_resource:	H
9cnn_block_8_batch_normalization_8_readvariableop_resource:	J
;cnn_block_8_batch_normalization_8_readvariableop_1_resource:	Y
Jcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	[
Lcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	
identity¢Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_6/batch_normalization_6/ReadVariableOp¢2cnn_block_6/batch_normalization_6/ReadVariableOp_1¢+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp¢*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp¢Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_7/batch_normalization_7/ReadVariableOp¢2cnn_block_7/batch_normalization_7/ReadVariableOp_1¢+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp¢*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp¢Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_8/batch_normalization_8/ReadVariableOp¢2cnn_block_8/batch_normalization_8/ReadVariableOp_1¢+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp¢*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp¢ conv2d_10/BiasAdd/ReadVariableOp¢conv2d_10/Conv2D/ReadVariableOp§
*cnn_block_6/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3cnn_block_6_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ê
cnn_block_6/conv2d_7/Conv2DConv2Dinput_tensor2cnn_block_6/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_6_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
cnn_block_6/conv2d_7/BiasAddBiasAdd$cnn_block_6/conv2d_7/Conv2D:output:03cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOp9cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype0«
2cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp;cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype0É
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Í
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
2cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%cnn_block_6/conv2d_7/BiasAdd:output:08cnn_block_6/batch_normalization_6/ReadVariableOp:value:0:cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0Icnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 
cnn_block_6/ReluRelu6cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
*cnn_block_7/conv2d_8/Conv2D/ReadVariableOpReadVariableOp3cnn_block_7_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
cnn_block_7/conv2d_8/Conv2DConv2Dcnn_block_6/Relu:activations:02cnn_block_7/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_7_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
cnn_block_7/conv2d_8/BiasAddBiasAdd$cnn_block_7/conv2d_8/Conv2D:output:03cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0cnn_block_7/batch_normalization_7/ReadVariableOpReadVariableOp9cnn_block_7_batch_normalization_7_readvariableop_resource*
_output_shapes	
:*
dtype0«
2cnn_block_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp;cnn_block_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:*
dtype0É
Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Í
Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
2cnn_block_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%cnn_block_7/conv2d_8/BiasAdd:output:08cnn_block_7/batch_normalization_7/ReadVariableOp:value:0:cnn_block_7/batch_normalization_7/ReadVariableOp_1:value:0Icnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 
cnn_block_7/ReluRelu6cnn_block_7/batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0´
conv2d_10/Conv2DConv2Dinput_tensor'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
addAddV2cnn_block_7/Relu:activations:0conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
*cnn_block_8/conv2d_9/Conv2D/ReadVariableOpReadVariableOp3cnn_block_8_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Å
cnn_block_8/conv2d_9/Conv2DConv2Dadd:z:02cnn_block_8/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_8_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
cnn_block_8/conv2d_9/BiasAddBiasAdd$cnn_block_8/conv2d_9/Conv2D:output:03cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0cnn_block_8/batch_normalization_8/ReadVariableOpReadVariableOp9cnn_block_8_batch_normalization_8_readvariableop_resource*
_output_shapes	
:*
dtype0«
2cnn_block_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp;cnn_block_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:*
dtype0É
Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Í
Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
2cnn_block_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3%cnn_block_8/conv2d_9/BiasAdd:output:08cnn_block_8/batch_normalization_8/ReadVariableOp:value:0:cnn_block_8/batch_normalization_8/ReadVariableOp_1:value:0Icnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 
cnn_block_8/ReluRelu6cnn_block_8/batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
max_pooling2d_1/MaxPoolMaxPoolcnn_block_8/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
x
IdentityIdentity max_pooling2d_1/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
NoOpNoOpB^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpD^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_11^cnn_block_6/batch_normalization_6/ReadVariableOp3^cnn_block_6/batch_normalization_6/ReadVariableOp_1,^cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp+^cnn_block_6/conv2d_7/Conv2D/ReadVariableOpB^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpD^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_11^cnn_block_7/batch_normalization_7/ReadVariableOp3^cnn_block_7/batch_normalization_7/ReadVariableOp_1,^cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp+^cnn_block_7/conv2d_8/Conv2D/ReadVariableOpB^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpD^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_11^cnn_block_8/batch_normalization_8/ReadVariableOp3^cnn_block_8/batch_normalization_8/ReadVariableOp_1,^cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp+^cnn_block_8/conv2d_9/Conv2D/ReadVariableOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ@: : : : : : : : : : : : : : : : : : : : 2
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpAcnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_6/batch_normalization_6/ReadVariableOp0cnn_block_6/batch_normalization_6/ReadVariableOp2h
2cnn_block_6/batch_normalization_6/ReadVariableOp_12cnn_block_6/batch_normalization_6/ReadVariableOp_12Z
+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp2X
*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp2
Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpAcnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_7/batch_normalization_7/ReadVariableOp0cnn_block_7/batch_normalization_7/ReadVariableOp2h
2cnn_block_7/batch_normalization_7/ReadVariableOp_12cnn_block_7/batch_normalization_7/ReadVariableOp_12Z
+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp2X
*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp2
Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpAcnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_8/batch_normalization_8/ReadVariableOp0cnn_block_8/batch_normalization_8/ReadVariableOp2h
2cnn_block_8/batch_normalization_8/ReadVariableOp_12cnn_block_8/batch_normalization_8/ReadVariableOp_12Z
+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp2X
*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
&
_user_specified_nameinput_tensor
Ê

O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_7124

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
	
Ï
4__inference_batch_normalization_3_layer_call_fn_9880

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
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
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_7124
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
	
Ð
5__inference_batch_normalization_5_layer_call_fn_10004

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
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
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_7252
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
¬ì
B
__inference__traced_save_10595
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableopD
@savev2_res_block_cnn_block_3_conv2d_3_kernel_read_readvariableopB
>savev2_res_block_cnn_block_3_conv2d_3_bias_read_readvariableopP
Lsavev2_res_block_cnn_block_3_batch_normalization_3_gamma_read_readvariableopO
Ksavev2_res_block_cnn_block_3_batch_normalization_3_beta_read_readvariableopD
@savev2_res_block_cnn_block_4_conv2d_4_kernel_read_readvariableopB
>savev2_res_block_cnn_block_4_conv2d_4_bias_read_readvariableopP
Lsavev2_res_block_cnn_block_4_batch_normalization_4_gamma_read_readvariableopO
Ksavev2_res_block_cnn_block_4_batch_normalization_4_beta_read_readvariableopD
@savev2_res_block_cnn_block_5_conv2d_5_kernel_read_readvariableopB
>savev2_res_block_cnn_block_5_conv2d_5_bias_read_readvariableopP
Lsavev2_res_block_cnn_block_5_batch_normalization_5_gamma_read_readvariableopO
Ksavev2_res_block_cnn_block_5_batch_normalization_5_beta_read_readvariableop8
4savev2_res_block_conv2d_6_kernel_read_readvariableop6
2savev2_res_block_conv2d_6_bias_read_readvariableopV
Rsavev2_res_block_cnn_block_3_batch_normalization_3_moving_mean_read_readvariableopZ
Vsavev2_res_block_cnn_block_3_batch_normalization_3_moving_variance_read_readvariableopV
Rsavev2_res_block_cnn_block_4_batch_normalization_4_moving_mean_read_readvariableopZ
Vsavev2_res_block_cnn_block_4_batch_normalization_4_moving_variance_read_readvariableopV
Rsavev2_res_block_cnn_block_5_batch_normalization_5_moving_mean_read_readvariableopZ
Vsavev2_res_block_cnn_block_5_batch_normalization_5_moving_variance_read_readvariableopF
Bsavev2_res_block_1_cnn_block_6_conv2d_7_kernel_read_readvariableopD
@savev2_res_block_1_cnn_block_6_conv2d_7_bias_read_readvariableopR
Nsavev2_res_block_1_cnn_block_6_batch_normalization_6_gamma_read_readvariableopQ
Msavev2_res_block_1_cnn_block_6_batch_normalization_6_beta_read_readvariableopF
Bsavev2_res_block_1_cnn_block_7_conv2d_8_kernel_read_readvariableopD
@savev2_res_block_1_cnn_block_7_conv2d_8_bias_read_readvariableopR
Nsavev2_res_block_1_cnn_block_7_batch_normalization_7_gamma_read_readvariableopQ
Msavev2_res_block_1_cnn_block_7_batch_normalization_7_beta_read_readvariableopF
Bsavev2_res_block_1_cnn_block_8_conv2d_9_kernel_read_readvariableopD
@savev2_res_block_1_cnn_block_8_conv2d_9_bias_read_readvariableopR
Nsavev2_res_block_1_cnn_block_8_batch_normalization_8_gamma_read_readvariableopQ
Msavev2_res_block_1_cnn_block_8_batch_normalization_8_beta_read_readvariableop;
7savev2_res_block_1_conv2d_10_kernel_read_readvariableop9
5savev2_res_block_1_conv2d_10_bias_read_readvariableopX
Tsavev2_res_block_1_cnn_block_6_batch_normalization_6_moving_mean_read_readvariableop\
Xsavev2_res_block_1_cnn_block_6_batch_normalization_6_moving_variance_read_readvariableopX
Tsavev2_res_block_1_cnn_block_7_batch_normalization_7_moving_mean_read_readvariableop\
Xsavev2_res_block_1_cnn_block_7_batch_normalization_7_moving_variance_read_readvariableopX
Tsavev2_res_block_1_cnn_block_8_batch_normalization_8_moving_mean_read_readvariableop\
Xsavev2_res_block_1_cnn_block_8_batch_normalization_8_moving_variance_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableopK
Gsavev2_adam_res_block_cnn_block_3_conv2d_3_kernel_m_read_readvariableopI
Esavev2_adam_res_block_cnn_block_3_conv2d_3_bias_m_read_readvariableopW
Ssavev2_adam_res_block_cnn_block_3_batch_normalization_3_gamma_m_read_readvariableopV
Rsavev2_adam_res_block_cnn_block_3_batch_normalization_3_beta_m_read_readvariableopK
Gsavev2_adam_res_block_cnn_block_4_conv2d_4_kernel_m_read_readvariableopI
Esavev2_adam_res_block_cnn_block_4_conv2d_4_bias_m_read_readvariableopW
Ssavev2_adam_res_block_cnn_block_4_batch_normalization_4_gamma_m_read_readvariableopV
Rsavev2_adam_res_block_cnn_block_4_batch_normalization_4_beta_m_read_readvariableopK
Gsavev2_adam_res_block_cnn_block_5_conv2d_5_kernel_m_read_readvariableopI
Esavev2_adam_res_block_cnn_block_5_conv2d_5_bias_m_read_readvariableopW
Ssavev2_adam_res_block_cnn_block_5_batch_normalization_5_gamma_m_read_readvariableopV
Rsavev2_adam_res_block_cnn_block_5_batch_normalization_5_beta_m_read_readvariableop?
;savev2_adam_res_block_conv2d_6_kernel_m_read_readvariableop=
9savev2_adam_res_block_conv2d_6_bias_m_read_readvariableopM
Isavev2_adam_res_block_1_cnn_block_6_conv2d_7_kernel_m_read_readvariableopK
Gsavev2_adam_res_block_1_cnn_block_6_conv2d_7_bias_m_read_readvariableopY
Usavev2_adam_res_block_1_cnn_block_6_batch_normalization_6_gamma_m_read_readvariableopX
Tsavev2_adam_res_block_1_cnn_block_6_batch_normalization_6_beta_m_read_readvariableopM
Isavev2_adam_res_block_1_cnn_block_7_conv2d_8_kernel_m_read_readvariableopK
Gsavev2_adam_res_block_1_cnn_block_7_conv2d_8_bias_m_read_readvariableopY
Usavev2_adam_res_block_1_cnn_block_7_batch_normalization_7_gamma_m_read_readvariableopX
Tsavev2_adam_res_block_1_cnn_block_7_batch_normalization_7_beta_m_read_readvariableopM
Isavev2_adam_res_block_1_cnn_block_8_conv2d_9_kernel_m_read_readvariableopK
Gsavev2_adam_res_block_1_cnn_block_8_conv2d_9_bias_m_read_readvariableopY
Usavev2_adam_res_block_1_cnn_block_8_batch_normalization_8_gamma_m_read_readvariableopX
Tsavev2_adam_res_block_1_cnn_block_8_batch_normalization_8_beta_m_read_readvariableopB
>savev2_adam_res_block_1_conv2d_10_kernel_m_read_readvariableop@
<savev2_adam_res_block_1_conv2d_10_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableopK
Gsavev2_adam_res_block_cnn_block_3_conv2d_3_kernel_v_read_readvariableopI
Esavev2_adam_res_block_cnn_block_3_conv2d_3_bias_v_read_readvariableopW
Ssavev2_adam_res_block_cnn_block_3_batch_normalization_3_gamma_v_read_readvariableopV
Rsavev2_adam_res_block_cnn_block_3_batch_normalization_3_beta_v_read_readvariableopK
Gsavev2_adam_res_block_cnn_block_4_conv2d_4_kernel_v_read_readvariableopI
Esavev2_adam_res_block_cnn_block_4_conv2d_4_bias_v_read_readvariableopW
Ssavev2_adam_res_block_cnn_block_4_batch_normalization_4_gamma_v_read_readvariableopV
Rsavev2_adam_res_block_cnn_block_4_batch_normalization_4_beta_v_read_readvariableopK
Gsavev2_adam_res_block_cnn_block_5_conv2d_5_kernel_v_read_readvariableopI
Esavev2_adam_res_block_cnn_block_5_conv2d_5_bias_v_read_readvariableopW
Ssavev2_adam_res_block_cnn_block_5_batch_normalization_5_gamma_v_read_readvariableopV
Rsavev2_adam_res_block_cnn_block_5_batch_normalization_5_beta_v_read_readvariableop?
;savev2_adam_res_block_conv2d_6_kernel_v_read_readvariableop=
9savev2_adam_res_block_conv2d_6_bias_v_read_readvariableopM
Isavev2_adam_res_block_1_cnn_block_6_conv2d_7_kernel_v_read_readvariableopK
Gsavev2_adam_res_block_1_cnn_block_6_conv2d_7_bias_v_read_readvariableopY
Usavev2_adam_res_block_1_cnn_block_6_batch_normalization_6_gamma_v_read_readvariableopX
Tsavev2_adam_res_block_1_cnn_block_6_batch_normalization_6_beta_v_read_readvariableopM
Isavev2_adam_res_block_1_cnn_block_7_conv2d_8_kernel_v_read_readvariableopK
Gsavev2_adam_res_block_1_cnn_block_7_conv2d_8_bias_v_read_readvariableopY
Usavev2_adam_res_block_1_cnn_block_7_batch_normalization_7_gamma_v_read_readvariableopX
Tsavev2_adam_res_block_1_cnn_block_7_batch_normalization_7_beta_v_read_readvariableopM
Isavev2_adam_res_block_1_cnn_block_8_conv2d_9_kernel_v_read_readvariableopK
Gsavev2_adam_res_block_1_cnn_block_8_conv2d_9_bias_v_read_readvariableopY
Usavev2_adam_res_block_1_cnn_block_8_batch_normalization_8_gamma_v_read_readvariableopX
Tsavev2_adam_res_block_1_cnn_block_8_batch_normalization_8_beta_v_read_readvariableopB
>savev2_adam_res_block_1_conv2d_10_kernel_v_read_readvariableop@
<savev2_adam_res_block_1_conv2d_10_bias_v_read_readvariableop
savev2_const

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
: ã2
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:p*
dtype0*2
value2Bÿ1pB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÐ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:p*
dtype0*õ
valueëBèpB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B @
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop@savev2_res_block_cnn_block_3_conv2d_3_kernel_read_readvariableop>savev2_res_block_cnn_block_3_conv2d_3_bias_read_readvariableopLsavev2_res_block_cnn_block_3_batch_normalization_3_gamma_read_readvariableopKsavev2_res_block_cnn_block_3_batch_normalization_3_beta_read_readvariableop@savev2_res_block_cnn_block_4_conv2d_4_kernel_read_readvariableop>savev2_res_block_cnn_block_4_conv2d_4_bias_read_readvariableopLsavev2_res_block_cnn_block_4_batch_normalization_4_gamma_read_readvariableopKsavev2_res_block_cnn_block_4_batch_normalization_4_beta_read_readvariableop@savev2_res_block_cnn_block_5_conv2d_5_kernel_read_readvariableop>savev2_res_block_cnn_block_5_conv2d_5_bias_read_readvariableopLsavev2_res_block_cnn_block_5_batch_normalization_5_gamma_read_readvariableopKsavev2_res_block_cnn_block_5_batch_normalization_5_beta_read_readvariableop4savev2_res_block_conv2d_6_kernel_read_readvariableop2savev2_res_block_conv2d_6_bias_read_readvariableopRsavev2_res_block_cnn_block_3_batch_normalization_3_moving_mean_read_readvariableopVsavev2_res_block_cnn_block_3_batch_normalization_3_moving_variance_read_readvariableopRsavev2_res_block_cnn_block_4_batch_normalization_4_moving_mean_read_readvariableopVsavev2_res_block_cnn_block_4_batch_normalization_4_moving_variance_read_readvariableopRsavev2_res_block_cnn_block_5_batch_normalization_5_moving_mean_read_readvariableopVsavev2_res_block_cnn_block_5_batch_normalization_5_moving_variance_read_readvariableopBsavev2_res_block_1_cnn_block_6_conv2d_7_kernel_read_readvariableop@savev2_res_block_1_cnn_block_6_conv2d_7_bias_read_readvariableopNsavev2_res_block_1_cnn_block_6_batch_normalization_6_gamma_read_readvariableopMsavev2_res_block_1_cnn_block_6_batch_normalization_6_beta_read_readvariableopBsavev2_res_block_1_cnn_block_7_conv2d_8_kernel_read_readvariableop@savev2_res_block_1_cnn_block_7_conv2d_8_bias_read_readvariableopNsavev2_res_block_1_cnn_block_7_batch_normalization_7_gamma_read_readvariableopMsavev2_res_block_1_cnn_block_7_batch_normalization_7_beta_read_readvariableopBsavev2_res_block_1_cnn_block_8_conv2d_9_kernel_read_readvariableop@savev2_res_block_1_cnn_block_8_conv2d_9_bias_read_readvariableopNsavev2_res_block_1_cnn_block_8_batch_normalization_8_gamma_read_readvariableopMsavev2_res_block_1_cnn_block_8_batch_normalization_8_beta_read_readvariableop7savev2_res_block_1_conv2d_10_kernel_read_readvariableop5savev2_res_block_1_conv2d_10_bias_read_readvariableopTsavev2_res_block_1_cnn_block_6_batch_normalization_6_moving_mean_read_readvariableopXsavev2_res_block_1_cnn_block_6_batch_normalization_6_moving_variance_read_readvariableopTsavev2_res_block_1_cnn_block_7_batch_normalization_7_moving_mean_read_readvariableopXsavev2_res_block_1_cnn_block_7_batch_normalization_7_moving_variance_read_readvariableopTsavev2_res_block_1_cnn_block_8_batch_normalization_8_moving_mean_read_readvariableopXsavev2_res_block_1_cnn_block_8_batch_normalization_8_moving_variance_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableopGsavev2_adam_res_block_cnn_block_3_conv2d_3_kernel_m_read_readvariableopEsavev2_adam_res_block_cnn_block_3_conv2d_3_bias_m_read_readvariableopSsavev2_adam_res_block_cnn_block_3_batch_normalization_3_gamma_m_read_readvariableopRsavev2_adam_res_block_cnn_block_3_batch_normalization_3_beta_m_read_readvariableopGsavev2_adam_res_block_cnn_block_4_conv2d_4_kernel_m_read_readvariableopEsavev2_adam_res_block_cnn_block_4_conv2d_4_bias_m_read_readvariableopSsavev2_adam_res_block_cnn_block_4_batch_normalization_4_gamma_m_read_readvariableopRsavev2_adam_res_block_cnn_block_4_batch_normalization_4_beta_m_read_readvariableopGsavev2_adam_res_block_cnn_block_5_conv2d_5_kernel_m_read_readvariableopEsavev2_adam_res_block_cnn_block_5_conv2d_5_bias_m_read_readvariableopSsavev2_adam_res_block_cnn_block_5_batch_normalization_5_gamma_m_read_readvariableopRsavev2_adam_res_block_cnn_block_5_batch_normalization_5_beta_m_read_readvariableop;savev2_adam_res_block_conv2d_6_kernel_m_read_readvariableop9savev2_adam_res_block_conv2d_6_bias_m_read_readvariableopIsavev2_adam_res_block_1_cnn_block_6_conv2d_7_kernel_m_read_readvariableopGsavev2_adam_res_block_1_cnn_block_6_conv2d_7_bias_m_read_readvariableopUsavev2_adam_res_block_1_cnn_block_6_batch_normalization_6_gamma_m_read_readvariableopTsavev2_adam_res_block_1_cnn_block_6_batch_normalization_6_beta_m_read_readvariableopIsavev2_adam_res_block_1_cnn_block_7_conv2d_8_kernel_m_read_readvariableopGsavev2_adam_res_block_1_cnn_block_7_conv2d_8_bias_m_read_readvariableopUsavev2_adam_res_block_1_cnn_block_7_batch_normalization_7_gamma_m_read_readvariableopTsavev2_adam_res_block_1_cnn_block_7_batch_normalization_7_beta_m_read_readvariableopIsavev2_adam_res_block_1_cnn_block_8_conv2d_9_kernel_m_read_readvariableopGsavev2_adam_res_block_1_cnn_block_8_conv2d_9_bias_m_read_readvariableopUsavev2_adam_res_block_1_cnn_block_8_batch_normalization_8_gamma_m_read_readvariableopTsavev2_adam_res_block_1_cnn_block_8_batch_normalization_8_beta_m_read_readvariableop>savev2_adam_res_block_1_conv2d_10_kernel_m_read_readvariableop<savev2_adam_res_block_1_conv2d_10_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopGsavev2_adam_res_block_cnn_block_3_conv2d_3_kernel_v_read_readvariableopEsavev2_adam_res_block_cnn_block_3_conv2d_3_bias_v_read_readvariableopSsavev2_adam_res_block_cnn_block_3_batch_normalization_3_gamma_v_read_readvariableopRsavev2_adam_res_block_cnn_block_3_batch_normalization_3_beta_v_read_readvariableopGsavev2_adam_res_block_cnn_block_4_conv2d_4_kernel_v_read_readvariableopEsavev2_adam_res_block_cnn_block_4_conv2d_4_bias_v_read_readvariableopSsavev2_adam_res_block_cnn_block_4_batch_normalization_4_gamma_v_read_readvariableopRsavev2_adam_res_block_cnn_block_4_batch_normalization_4_beta_v_read_readvariableopGsavev2_adam_res_block_cnn_block_5_conv2d_5_kernel_v_read_readvariableopEsavev2_adam_res_block_cnn_block_5_conv2d_5_bias_v_read_readvariableopSsavev2_adam_res_block_cnn_block_5_batch_normalization_5_gamma_v_read_readvariableopRsavev2_adam_res_block_cnn_block_5_batch_normalization_5_beta_v_read_readvariableop;savev2_adam_res_block_conv2d_6_kernel_v_read_readvariableop9savev2_adam_res_block_conv2d_6_bias_v_read_readvariableopIsavev2_adam_res_block_1_cnn_block_6_conv2d_7_kernel_v_read_readvariableopGsavev2_adam_res_block_1_cnn_block_6_conv2d_7_bias_v_read_readvariableopUsavev2_adam_res_block_1_cnn_block_6_batch_normalization_6_gamma_v_read_readvariableopTsavev2_adam_res_block_1_cnn_block_6_batch_normalization_6_beta_v_read_readvariableopIsavev2_adam_res_block_1_cnn_block_7_conv2d_8_kernel_v_read_readvariableopGsavev2_adam_res_block_1_cnn_block_7_conv2d_8_bias_v_read_readvariableopUsavev2_adam_res_block_1_cnn_block_7_batch_normalization_7_gamma_v_read_readvariableopTsavev2_adam_res_block_1_cnn_block_7_batch_normalization_7_beta_v_read_readvariableopIsavev2_adam_res_block_1_cnn_block_8_conv2d_9_kernel_v_read_readvariableopGsavev2_adam_res_block_1_cnn_block_8_conv2d_9_bias_v_read_readvariableopUsavev2_adam_res_block_1_cnn_block_8_batch_normalization_8_gamma_v_read_readvariableopTsavev2_adam_res_block_1_cnn_block_8_batch_normalization_8_beta_v_read_readvariableop>savev2_adam_res_block_1_conv2d_10_kernel_v_read_readvariableop<savev2_adam_res_block_1_conv2d_10_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *~
dtypest
r2p	
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

identity_1Identity_1:output:0*ô
_input_shapesâ
ß: :	b
:
: : : : :  : : : : @:@:@:@: : : : : : :@:@:@::::::::::::@:::::::: : : : : : : : : :	b
:
: : : : :  : : : : @:@:@:@: : :@::::::::::::@::	b
:
: : : : :  : : : : @:@:@:@: : :@::::::::::::@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	b
: 

_output_shapes
:
:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::! 

_output_shapes	
::!!

_output_shapes	
::!"

_output_shapes	
::-#)
'
_output_shapes
:@:!$

_output_shapes	
::!%

_output_shapes	
::!&

_output_shapes	
::!'

_output_shapes	
::!(

_output_shapes	
::!)

_output_shapes	
::!*

_output_shapes	
::+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :%4!

_output_shapes
:	b
: 5

_output_shapes
:
:,6(
&
_output_shapes
: : 7

_output_shapes
: : 8

_output_shapes
: : 9

_output_shapes
: :,:(
&
_output_shapes
:  : ;

_output_shapes
: : <

_output_shapes
: : =

_output_shapes
: :,>(
&
_output_shapes
: @: ?

_output_shapes
:@: @

_output_shapes
:@: A

_output_shapes
:@:,B(
&
_output_shapes
: : C

_output_shapes
: :-D)
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
::.H*
(
_output_shapes
::!I

_output_shapes	
::!J

_output_shapes	
::!K

_output_shapes	
::.L*
(
_output_shapes
::!M

_output_shapes	
::!N

_output_shapes	
::!O

_output_shapes	
::-P)
'
_output_shapes
:@:!Q

_output_shapes	
::%R!

_output_shapes
:	b
: S

_output_shapes
:
:,T(
&
_output_shapes
: : U

_output_shapes
: : V

_output_shapes
: : W

_output_shapes
: :,X(
&
_output_shapes
:  : Y

_output_shapes
: : Z

_output_shapes
: : [

_output_shapes
: :,\(
&
_output_shapes
: @: ]

_output_shapes
:@: ^

_output_shapes
:@: _

_output_shapes
:@:,`(
&
_output_shapes
: : a

_output_shapes
: :-b)
'
_output_shapes
:@:!c

_output_shapes	
::!d

_output_shapes	
::!e

_output_shapes	
::.f*
(
_output_shapes
::!g

_output_shapes	
::!h

_output_shapes	
::!i

_output_shapes	
::.j*
(
_output_shapes
::!k

_output_shapes	
::!l

_output_shapes	
::!m

_output_shapes	
::-n)
'
_output_shapes
:@:!o

_output_shapes	
::p

_output_shapes
: 
Ê


&__inference_model_1_layer_call_fn_8570
input_1!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: $

unknown_13: @

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@%

unknown_19:@

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:	

unknown_24:	&

unknown_25:

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	%

unknown_31:@

unknown_32:	&

unknown_33:

unknown_34:	

unknown_35:	

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:	b


unknown_40:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_8394o
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
_construction_contextkEagerRuntime*
_input_shapesq
o:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ý
2
A__inference_model_1_layer_call_and_return_conditional_losses_9337

inputsW
=res_block_cnn_block_3_conv2d_3_conv2d_readvariableop_resource: L
>res_block_cnn_block_3_conv2d_3_biasadd_readvariableop_resource: Q
Cres_block_cnn_block_3_batch_normalization_3_readvariableop_resource: S
Eres_block_cnn_block_3_batch_normalization_3_readvariableop_1_resource: b
Tres_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource: d
Vres_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: W
=res_block_cnn_block_4_conv2d_4_conv2d_readvariableop_resource:  L
>res_block_cnn_block_4_conv2d_4_biasadd_readvariableop_resource: Q
Cres_block_cnn_block_4_batch_normalization_4_readvariableop_resource: S
Eres_block_cnn_block_4_batch_normalization_4_readvariableop_1_resource: b
Tres_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource: d
Vres_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: K
1res_block_conv2d_6_conv2d_readvariableop_resource: @
2res_block_conv2d_6_biasadd_readvariableop_resource: W
=res_block_cnn_block_5_conv2d_5_conv2d_readvariableop_resource: @L
>res_block_cnn_block_5_conv2d_5_biasadd_readvariableop_resource:@Q
Cres_block_cnn_block_5_batch_normalization_5_readvariableop_resource:@S
Eres_block_cnn_block_5_batch_normalization_5_readvariableop_1_resource:@b
Tres_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@d
Vres_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@Z
?res_block_1_cnn_block_6_conv2d_7_conv2d_readvariableop_resource:@O
@res_block_1_cnn_block_6_conv2d_7_biasadd_readvariableop_resource:	T
Eres_block_1_cnn_block_6_batch_normalization_6_readvariableop_resource:	V
Gres_block_1_cnn_block_6_batch_normalization_6_readvariableop_1_resource:	e
Vres_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	g
Xres_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	[
?res_block_1_cnn_block_7_conv2d_8_conv2d_readvariableop_resource:O
@res_block_1_cnn_block_7_conv2d_8_biasadd_readvariableop_resource:	T
Eres_block_1_cnn_block_7_batch_normalization_7_readvariableop_resource:	V
Gres_block_1_cnn_block_7_batch_normalization_7_readvariableop_1_resource:	e
Vres_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	g
Xres_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	O
4res_block_1_conv2d_10_conv2d_readvariableop_resource:@D
5res_block_1_conv2d_10_biasadd_readvariableop_resource:	[
?res_block_1_cnn_block_8_conv2d_9_conv2d_readvariableop_resource:O
@res_block_1_cnn_block_8_conv2d_9_biasadd_readvariableop_resource:	T
Eres_block_1_cnn_block_8_batch_normalization_8_readvariableop_resource:	V
Gres_block_1_cnn_block_8_batch_normalization_8_readvariableop_1_resource:	e
Vres_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	g
Xres_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	9
&dense_2_matmul_readvariableop_resource:	b
5
'dense_2_biasadd_readvariableop_resource:

identity¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢Kres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢Mres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢:res_block/cnn_block_3/batch_normalization_3/ReadVariableOp¢<res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1¢5res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp¢4res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp¢Kres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp¢Mres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1¢:res_block/cnn_block_4/batch_normalization_4/ReadVariableOp¢<res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1¢5res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp¢4res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp¢Kres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp¢Mres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1¢:res_block/cnn_block_5/batch_normalization_5/ReadVariableOp¢<res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1¢5res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp¢4res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp¢)res_block/conv2d_6/BiasAdd/ReadVariableOp¢(res_block/conv2d_6/Conv2D/ReadVariableOp¢Mres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp¢Ores_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1¢<res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp¢>res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1¢7res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp¢6res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp¢Mres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp¢Ores_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1¢<res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp¢>res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1¢7res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp¢6res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp¢Mres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢Ores_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢<res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp¢>res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1¢7res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp¢6res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp¢,res_block_1/conv2d_10/BiasAdd/ReadVariableOp¢+res_block_1/conv2d_10/Conv2D/ReadVariableOpº
4res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp=res_block_cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0×
%res_block/cnn_block_3/conv2d_3/Conv2DConv2Dinputs<res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
°
5res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp>res_block_cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ú
&res_block/cnn_block_3/conv2d_3/BiasAddBiasAdd.res_block/cnn_block_3/conv2d_3/Conv2D:output:0=res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ º
:res_block/cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOpCres_block_cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype0¾
<res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOpEres_block_cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ü
Kres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpTres_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0à
Mres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVres_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0»
<res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3/res_block/cnn_block_3/conv2d_3/BiasAdd:output:0Bres_block/cnn_block_3/batch_normalization_3/ReadVariableOp:value:0Dres_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0Sres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Ures_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 
res_block/cnn_block_3/ReluRelu@res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ º
4res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp=res_block_cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0ù
%res_block/cnn_block_4/conv2d_4/Conv2DConv2D(res_block/cnn_block_3/Relu:activations:0<res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
°
5res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp>res_block_cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ú
&res_block/cnn_block_4/conv2d_4/BiasAddBiasAdd.res_block/cnn_block_4/conv2d_4/Conv2D:output:0=res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ º
:res_block/cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOpCres_block_cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype0¾
<res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOpEres_block_cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype0Ü
Kres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpTres_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0à
Mres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVres_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0»
<res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3/res_block/cnn_block_4/conv2d_4/BiasAdd:output:0Bres_block/cnn_block_4/batch_normalization_4/ReadVariableOp:value:0Dres_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0Sres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Ures_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 
res_block/cnn_block_4/ReluRelu@res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¢
(res_block/conv2d_6/Conv2D/ReadVariableOpReadVariableOp1res_block_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0¿
res_block/conv2d_6/Conv2DConv2Dinputs0res_block/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

)res_block/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp2res_block_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¶
res_block/conv2d_6/BiasAddBiasAdd"res_block/conv2d_6/Conv2D:output:01res_block/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
res_block/addAddV2(res_block/cnn_block_4/Relu:activations:0#res_block/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ º
4res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp=res_block_cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0â
%res_block/cnn_block_5/conv2d_5/Conv2DConv2Dres_block/add:z:0<res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
°
5res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp>res_block_cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ú
&res_block/cnn_block_5/conv2d_5/BiasAddBiasAdd.res_block/cnn_block_5/conv2d_5/Conv2D:output:0=res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
:res_block/cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOpCres_block_cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0¾
<res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOpEres_block_cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ü
Kres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpTres_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0à
Mres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVres_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
<res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3/res_block/cnn_block_5/conv2d_5/BiasAdd:output:0Bres_block/cnn_block_5/batch_normalization_5/ReadVariableOp:value:0Dres_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0Sres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Ures_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 
res_block/cnn_block_5/ReluRelu@res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Á
res_block/max_pooling2d/MaxPoolMaxPool(res_block/cnn_block_5/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
¿
6res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOpReadVariableOp?res_block_1_cnn_block_6_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0þ
'res_block_1/cnn_block_6/conv2d_7/Conv2DConv2D(res_block/max_pooling2d/MaxPool:output:0>res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
µ
7res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp@res_block_1_cnn_block_6_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0á
(res_block_1/cnn_block_6/conv2d_7/BiasAddBiasAdd0res_block_1/cnn_block_6/conv2d_7/Conv2D:output:0?res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
<res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOpEres_block_1_cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
>res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOpGres_block_1_cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype0á
Mres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpVres_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0å
Ores_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpXres_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ì
>res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV31res_block_1/cnn_block_6/conv2d_7/BiasAdd:output:0Dres_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp:value:0Fres_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0Ures_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Wres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( £
res_block_1/cnn_block_6/ReluReluBres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
6res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOpReadVariableOp?res_block_1_cnn_block_7_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
'res_block_1/cnn_block_7/conv2d_8/Conv2DConv2D*res_block_1/cnn_block_6/Relu:activations:0>res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
µ
7res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp@res_block_1_cnn_block_7_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0á
(res_block_1/cnn_block_7/conv2d_8/BiasAddBiasAdd0res_block_1/cnn_block_7/conv2d_8/Conv2D:output:0?res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
<res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOpReadVariableOpEres_block_1_cnn_block_7_batch_normalization_7_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
>res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1ReadVariableOpGres_block_1_cnn_block_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:*
dtype0á
Mres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpVres_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0å
Ores_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpXres_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ì
>res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV31res_block_1/cnn_block_7/conv2d_8/BiasAdd:output:0Dres_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp:value:0Fres_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1:value:0Ures_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Wres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( £
res_block_1/cnn_block_7/ReluReluBres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
+res_block_1/conv2d_10/Conv2D/ReadVariableOpReadVariableOp4res_block_1_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0è
res_block_1/conv2d_10/Conv2DConv2D(res_block/max_pooling2d/MaxPool:output:03res_block_1/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

,res_block_1/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp5res_block_1_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0À
res_block_1/conv2d_10/BiasAddBiasAdd%res_block_1/conv2d_10/Conv2D:output:04res_block_1/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
res_block_1/addAddV2*res_block_1/cnn_block_7/Relu:activations:0&res_block_1/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
6res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOpReadVariableOp?res_block_1_cnn_block_8_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0é
'res_block_1/cnn_block_8/conv2d_9/Conv2DConv2Dres_block_1/add:z:0>res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
µ
7res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp@res_block_1_cnn_block_8_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0á
(res_block_1/cnn_block_8/conv2d_9/BiasAddBiasAdd0res_block_1/cnn_block_8/conv2d_9/Conv2D:output:0?res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
<res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOpReadVariableOpEres_block_1_cnn_block_8_batch_normalization_8_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
>res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1ReadVariableOpGres_block_1_cnn_block_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:*
dtype0á
Mres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpVres_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0å
Ores_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpXres_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ì
>res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV31res_block_1/cnn_block_8/conv2d_9/BiasAdd:output:0Dres_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp:value:0Fres_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1:value:0Ures_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Wres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( £
res_block_1/cnn_block_8/ReluReluBres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
#res_block_1/max_pooling2d_1/MaxPoolMaxPool*res_block_1/cnn_block_8/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 1  
flatten_1/ReshapeReshape,res_block_1/max_pooling2d_1/MaxPool:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	b
*
dtype0
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOpL^res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpN^res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1;^res_block/cnn_block_3/batch_normalization_3/ReadVariableOp=^res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_16^res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp5^res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpL^res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpN^res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1;^res_block/cnn_block_4/batch_normalization_4/ReadVariableOp=^res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_16^res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp5^res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpL^res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpN^res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1;^res_block/cnn_block_5/batch_normalization_5/ReadVariableOp=^res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_16^res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp5^res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp*^res_block/conv2d_6/BiasAdd/ReadVariableOp)^res_block/conv2d_6/Conv2D/ReadVariableOpN^res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpP^res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1=^res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp?^res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_18^res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp7^res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOpN^res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpP^res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1=^res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp?^res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_18^res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp7^res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOpN^res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpP^res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1=^res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp?^res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_18^res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp7^res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp-^res_block_1/conv2d_10/BiasAdd/ReadVariableOp,^res_block_1/conv2d_10/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesq
o:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2
Kres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpKres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2
Mres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Mres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12x
:res_block/cnn_block_3/batch_normalization_3/ReadVariableOp:res_block/cnn_block_3/batch_normalization_3/ReadVariableOp2|
<res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1<res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_12n
5res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp5res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp2l
4res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp4res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp2
Kres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpKres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2
Mres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Mres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12x
:res_block/cnn_block_4/batch_normalization_4/ReadVariableOp:res_block/cnn_block_4/batch_normalization_4/ReadVariableOp2|
<res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1<res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_12n
5res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp5res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp2l
4res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp4res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp2
Kres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpKres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2
Mres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Mres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12x
:res_block/cnn_block_5/batch_normalization_5/ReadVariableOp:res_block/cnn_block_5/batch_normalization_5/ReadVariableOp2|
<res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1<res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_12n
5res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp5res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp2l
4res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp4res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp2V
)res_block/conv2d_6/BiasAdd/ReadVariableOp)res_block/conv2d_6/BiasAdd/ReadVariableOp2T
(res_block/conv2d_6/Conv2D/ReadVariableOp(res_block/conv2d_6/Conv2D/ReadVariableOp2
Mres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpMres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2¢
Ores_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ores_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12|
<res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp<res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp2
>res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1>res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_12r
7res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp7res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp2p
6res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp6res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp2
Mres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpMres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2¢
Ores_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ores_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12|
<res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp<res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp2
>res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1>res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_12r
7res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp7res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp2p
6res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp6res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp2
Mres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpMres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2¢
Ores_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ores_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12|
<res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp<res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp2
>res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1>res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_12r
7res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp7res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp2p
6res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp6res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp2\
,res_block_1/conv2d_10/BiasAdd/ReadVariableOp,res_block_1/conv2d_10/BiasAdd/ReadVariableOp2Z
+res_block_1/conv2d_10/Conv2D/ReadVariableOp+res_block_1/conv2d_10/Conv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7303

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û

P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10221

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê

O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_7252

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


(__inference_res_block_layer_call_fn_9382
input_tensor!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: $

unknown_13: @

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_res_block_layer_call_and_return_conditional_losses_7592w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_tensor

Ã
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10115

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
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
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
Ç


&__inference_model_1_layer_call_fn_8940

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: $

unknown_13: @

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@%

unknown_19:@

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:	

unknown_24:	&

unknown_25:

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	%

unknown_31:@

unknown_32:	&

unknown_33:

unknown_34:	

unknown_35:	

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:	b


unknown_40:

identity¢StatefulPartitionedCall
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
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_7776o
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
_construction_contextkEagerRuntime*
_input_shapesq
o:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç


&__inference_model_1_layer_call_fn_9029

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: $

unknown_13: @

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@%

unknown_19:@

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:	

unknown_24:	&

unknown_25:

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	%

unknown_31:@

unknown_32:	&

unknown_33:

unknown_34:	

unknown_35:	

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:	b


unknown_40:

identity¢StatefulPartitionedCall
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
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_8394o
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
_construction_contextkEagerRuntime*
_input_shapesq
o:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_9867

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
k
¤
C__inference_res_block_layer_call_and_return_conditional_losses_7592
input_tensorM
3cnn_block_3_conv2d_3_conv2d_readvariableop_resource: B
4cnn_block_3_conv2d_3_biasadd_readvariableop_resource: G
9cnn_block_3_batch_normalization_3_readvariableop_resource: I
;cnn_block_3_batch_normalization_3_readvariableop_1_resource: X
Jcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource: Z
Lcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: M
3cnn_block_4_conv2d_4_conv2d_readvariableop_resource:  B
4cnn_block_4_conv2d_4_biasadd_readvariableop_resource: G
9cnn_block_4_batch_normalization_4_readvariableop_resource: I
;cnn_block_4_batch_normalization_4_readvariableop_1_resource: X
Jcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource: Z
Lcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: M
3cnn_block_5_conv2d_5_conv2d_readvariableop_resource: @B
4cnn_block_5_conv2d_5_biasadd_readvariableop_resource:@G
9cnn_block_5_batch_normalization_5_readvariableop_resource:@I
;cnn_block_5_batch_normalization_5_readvariableop_1_resource:@X
Jcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@Z
Lcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@
identity¢Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_3/batch_normalization_3/ReadVariableOp¢2cnn_block_3/batch_normalization_3/ReadVariableOp_1¢+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp¢*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp¢Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_4/batch_normalization_4/ReadVariableOp¢2cnn_block_4/batch_normalization_4/ReadVariableOp_1¢+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp¢*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp¢Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_5/batch_normalization_5/ReadVariableOp¢2cnn_block_5/batch_normalization_5/ReadVariableOp_1¢+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp¢*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp¢conv2d_6/BiasAdd/ReadVariableOp¢conv2d_6/Conv2D/ReadVariableOp¦
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0É
cnn_block_3/conv2d_3/Conv2DConv2Dinput_tensor2cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¼
cnn_block_3/conv2d_3/BiasAddBiasAdd$cnn_block_3/conv2d_3/Conv2D:output:03cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
0cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOp9cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype0ª
2cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp;cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ì
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ÿ
2cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%cnn_block_3/conv2d_3/BiasAdd:output:08cnn_block_3/batch_normalization_3/ReadVariableOp:value:0:cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0Icnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 
cnn_block_3/ReluRelu6cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Û
cnn_block_4/conv2d_4/Conv2DConv2Dcnn_block_3/Relu:activations:02cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¼
cnn_block_4/conv2d_4/BiasAddBiasAdd$cnn_block_4/conv2d_4/Conv2D:output:03cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
0cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOp9cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype0ª
2cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp;cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype0È
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ì
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ÿ
2cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%cnn_block_4/conv2d_4/BiasAdd:output:08cnn_block_4/batch_normalization_4/ReadVariableOp:value:0:cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0Icnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 
cnn_block_4/ReluRelu6cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0±
conv2d_6/Conv2DConv2Dinput_tensor&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
addAddV2cnn_block_4/Relu:activations:0conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ä
cnn_block_5/conv2d_5/Conv2DConv2Dadd:z:02cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¼
cnn_block_5/conv2d_5/BiasAddBiasAdd$cnn_block_5/conv2d_5/Conv2D:output:03cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
0cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOp9cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0ª
2cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp;cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ì
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ÿ
2cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%cnn_block_5/conv2d_5/BiasAdd:output:08cnn_block_5/batch_normalization_5/ReadVariableOp:value:0:cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0Icnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 
cnn_block_5/ReluRelu6cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@­
max_pooling2d/MaxPoolMaxPoolcnn_block_5/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
u
IdentityIdentitymax_pooling2d/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ð
NoOpNoOpB^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpD^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_11^cnn_block_3/batch_normalization_3/ReadVariableOp3^cnn_block_3/batch_normalization_3/ReadVariableOp_1,^cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp+^cnn_block_3/conv2d_3/Conv2D/ReadVariableOpB^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpD^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_11^cnn_block_4/batch_normalization_4/ReadVariableOp3^cnn_block_4/batch_normalization_4/ReadVariableOp_1,^cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp+^cnn_block_4/conv2d_4/Conv2D/ReadVariableOpB^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpD^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_11^cnn_block_5/batch_normalization_5/ReadVariableOp3^cnn_block_5/batch_normalization_5/ReadVariableOp_1,^cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp+^cnn_block_5/conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpAcnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_3/batch_normalization_3/ReadVariableOp0cnn_block_3/batch_normalization_3/ReadVariableOp2h
2cnn_block_3/batch_normalization_3/ReadVariableOp_12cnn_block_3/batch_normalization_3/ReadVariableOp_12Z
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp2X
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp2
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpAcnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_4/batch_normalization_4/ReadVariableOp0cnn_block_4/batch_normalization_4/ReadVariableOp2h
2cnn_block_4/batch_normalization_4/ReadVariableOp_12cnn_block_4/batch_normalization_4/ReadVariableOp_12Z
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp2X
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp2
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpAcnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_5/batch_normalization_5/ReadVariableOp0cnn_block_5/batch_normalization_5/ReadVariableOp2h
2cnn_block_5/batch_normalization_5/ReadVariableOp_12cnn_block_5/batch_normalization_5/ReadVariableOp_12Z
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp2X
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_tensor
Ê

O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_7188

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
¡
ª
*__inference_res_block_1_layer_call_fn_9622
input_tensor"
unknown:@
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	%

unknown_11:@

unknown_12:	&

unknown_13:

unknown_14:	

unknown_15:	

unknown_16:	

unknown_17:	

unknown_18:	
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_res_block_1_layer_call_and_return_conditional_losses_7709x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ@: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
&
_user_specified_nameinput_tensor
¤


"__inference_signature_wrapper_8851
input_1!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12: $

unknown_13: @

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@%

unknown_19:@

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:	

unknown_24:	&

unknown_25:

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	%

unknown_31:@

unknown_32:	&

unknown_33:

unknown_34:	

unknown_35:	

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:	b


unknown_40:

identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**2
config_proto" 

CPU

GPU2 *0J 8 *(
f#R!
__inference__wrapped_model_7102o
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
_construction_contextkEagerRuntime*
_input_shapesq
o:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

Ü
C__inference_res_block_layer_call_and_return_conditional_losses_9577
input_tensorM
3cnn_block_3_conv2d_3_conv2d_readvariableop_resource: B
4cnn_block_3_conv2d_3_biasadd_readvariableop_resource: G
9cnn_block_3_batch_normalization_3_readvariableop_resource: I
;cnn_block_3_batch_normalization_3_readvariableop_1_resource: X
Jcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource: Z
Lcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: M
3cnn_block_4_conv2d_4_conv2d_readvariableop_resource:  B
4cnn_block_4_conv2d_4_biasadd_readvariableop_resource: G
9cnn_block_4_batch_normalization_4_readvariableop_resource: I
;cnn_block_4_batch_normalization_4_readvariableop_1_resource: X
Jcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource: Z
Lcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: M
3cnn_block_5_conv2d_5_conv2d_readvariableop_resource: @B
4cnn_block_5_conv2d_5_biasadd_readvariableop_resource:@G
9cnn_block_5_batch_normalization_5_readvariableop_resource:@I
;cnn_block_5_batch_normalization_5_readvariableop_1_resource:@X
Jcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@Z
Lcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@
identity¢0cnn_block_3/batch_normalization_3/AssignNewValue¢2cnn_block_3/batch_normalization_3/AssignNewValue_1¢Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_3/batch_normalization_3/ReadVariableOp¢2cnn_block_3/batch_normalization_3/ReadVariableOp_1¢+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp¢*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp¢0cnn_block_4/batch_normalization_4/AssignNewValue¢2cnn_block_4/batch_normalization_4/AssignNewValue_1¢Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_4/batch_normalization_4/ReadVariableOp¢2cnn_block_4/batch_normalization_4/ReadVariableOp_1¢+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp¢*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp¢0cnn_block_5/batch_normalization_5/AssignNewValue¢2cnn_block_5/batch_normalization_5/AssignNewValue_1¢Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp¢Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1¢0cnn_block_5/batch_normalization_5/ReadVariableOp¢2cnn_block_5/batch_normalization_5/ReadVariableOp_1¢+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp¢*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp¢conv2d_6/BiasAdd/ReadVariableOp¢conv2d_6/Conv2D/ReadVariableOp¦
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0É
cnn_block_3/conv2d_3/Conv2DConv2Dinput_tensor2cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¼
cnn_block_3/conv2d_3/BiasAddBiasAdd$cnn_block_3/conv2d_3/Conv2D:output:03cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
0cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOp9cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype0ª
2cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp;cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ì
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0
2cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%cnn_block_3/conv2d_3/BiasAdd:output:08cnn_block_3/batch_normalization_3/ReadVariableOp:value:0:cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0Icnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<Î
0cnn_block_3/batch_normalization_3/AssignNewValueAssignVariableOpJcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource?cnn_block_3/batch_normalization_3/FusedBatchNormV3:batch_mean:0B^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ø
2cnn_block_3/batch_normalization_3/AssignNewValue_1AssignVariableOpLcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_3/batch_normalization_3/FusedBatchNormV3:batch_variance:0D^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
cnn_block_3/ReluRelu6cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Û
cnn_block_4/conv2d_4/Conv2DConv2Dcnn_block_3/Relu:activations:02cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¼
cnn_block_4/conv2d_4/BiasAddBiasAdd$cnn_block_4/conv2d_4/Conv2D:output:03cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
0cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOp9cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype0ª
2cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp;cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype0È
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ì
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0
2cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%cnn_block_4/conv2d_4/BiasAdd:output:08cnn_block_4/batch_normalization_4/ReadVariableOp:value:0:cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0Icnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<Î
0cnn_block_4/batch_normalization_4/AssignNewValueAssignVariableOpJcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource?cnn_block_4/batch_normalization_4/FusedBatchNormV3:batch_mean:0B^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ø
2cnn_block_4/batch_normalization_4/AssignNewValue_1AssignVariableOpLcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_4/batch_normalization_4/FusedBatchNormV3:batch_variance:0D^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
cnn_block_4/ReluRelu6cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0±
conv2d_6/Conv2DConv2Dinput_tensor&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
addAddV2cnn_block_4/Relu:activations:0conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ä
cnn_block_5/conv2d_5/Conv2DConv2Dadd:z:02cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¼
cnn_block_5/conv2d_5/BiasAddBiasAdd$cnn_block_5/conv2d_5/Conv2D:output:03cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
0cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOp9cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0ª
2cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp;cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ì
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
2cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%cnn_block_5/conv2d_5/BiasAdd:output:08cnn_block_5/batch_normalization_5/ReadVariableOp:value:0:cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0Icnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<Î
0cnn_block_5/batch_normalization_5/AssignNewValueAssignVariableOpJcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource?cnn_block_5/batch_normalization_5/FusedBatchNormV3:batch_mean:0B^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ø
2cnn_block_5/batch_normalization_5/AssignNewValue_1AssignVariableOpLcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_5/batch_normalization_5/FusedBatchNormV3:batch_variance:0D^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
cnn_block_5/ReluRelu6cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@­
max_pooling2d/MaxPoolMaxPoolcnn_block_5/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
u
IdentityIdentitymax_pooling2d/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¨
NoOpNoOp1^cnn_block_3/batch_normalization_3/AssignNewValue3^cnn_block_3/batch_normalization_3/AssignNewValue_1B^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpD^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_11^cnn_block_3/batch_normalization_3/ReadVariableOp3^cnn_block_3/batch_normalization_3/ReadVariableOp_1,^cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp+^cnn_block_3/conv2d_3/Conv2D/ReadVariableOp1^cnn_block_4/batch_normalization_4/AssignNewValue3^cnn_block_4/batch_normalization_4/AssignNewValue_1B^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpD^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_11^cnn_block_4/batch_normalization_4/ReadVariableOp3^cnn_block_4/batch_normalization_4/ReadVariableOp_1,^cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp+^cnn_block_4/conv2d_4/Conv2D/ReadVariableOp1^cnn_block_5/batch_normalization_5/AssignNewValue3^cnn_block_5/batch_normalization_5/AssignNewValue_1B^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpD^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_11^cnn_block_5/batch_normalization_5/ReadVariableOp3^cnn_block_5/batch_normalization_5/ReadVariableOp_1,^cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp+^cnn_block_5/conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2d
0cnn_block_3/batch_normalization_3/AssignNewValue0cnn_block_3/batch_normalization_3/AssignNewValue2h
2cnn_block_3/batch_normalization_3/AssignNewValue_12cnn_block_3/batch_normalization_3/AssignNewValue_12
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpAcnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_3/batch_normalization_3/ReadVariableOp0cnn_block_3/batch_normalization_3/ReadVariableOp2h
2cnn_block_3/batch_normalization_3/ReadVariableOp_12cnn_block_3/batch_normalization_3/ReadVariableOp_12Z
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp2X
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp2d
0cnn_block_4/batch_normalization_4/AssignNewValue0cnn_block_4/batch_normalization_4/AssignNewValue2h
2cnn_block_4/batch_normalization_4/AssignNewValue_12cnn_block_4/batch_normalization_4/AssignNewValue_12
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpAcnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_4/batch_normalization_4/ReadVariableOp0cnn_block_4/batch_normalization_4/ReadVariableOp2h
2cnn_block_4/batch_normalization_4/ReadVariableOp_12cnn_block_4/batch_normalization_4/ReadVariableOp_12Z
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp2X
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp2d
0cnn_block_5/batch_normalization_5/AssignNewValue0cnn_block_5/batch_normalization_5/AssignNewValue2h
2cnn_block_5/batch_normalization_5/AssignNewValue_12cnn_block_5/batch_normalization_5/AssignNewValue_12
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpAcnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_5/batch_normalization_5/ReadVariableOp0cnn_block_5/batch_normalization_5/ReadVariableOp2h
2cnn_block_5/batch_normalization_5/ReadVariableOp_12cnn_block_5/batch_normalization_5/ReadVariableOp_12Z
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp2X
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_tensor
ºÕ
Z
!__inference__traced_restore_10938
file_prefix2
assignvariableop_dense_2_kernel:	b
-
assignvariableop_1_dense_2_bias:
R
8assignvariableop_2_res_block_cnn_block_3_conv2d_3_kernel: D
6assignvariableop_3_res_block_cnn_block_3_conv2d_3_bias: R
Dassignvariableop_4_res_block_cnn_block_3_batch_normalization_3_gamma: Q
Cassignvariableop_5_res_block_cnn_block_3_batch_normalization_3_beta: R
8assignvariableop_6_res_block_cnn_block_4_conv2d_4_kernel:  D
6assignvariableop_7_res_block_cnn_block_4_conv2d_4_bias: R
Dassignvariableop_8_res_block_cnn_block_4_batch_normalization_4_gamma: Q
Cassignvariableop_9_res_block_cnn_block_4_batch_normalization_4_beta: S
9assignvariableop_10_res_block_cnn_block_5_conv2d_5_kernel: @E
7assignvariableop_11_res_block_cnn_block_5_conv2d_5_bias:@S
Eassignvariableop_12_res_block_cnn_block_5_batch_normalization_5_gamma:@R
Dassignvariableop_13_res_block_cnn_block_5_batch_normalization_5_beta:@G
-assignvariableop_14_res_block_conv2d_6_kernel: 9
+assignvariableop_15_res_block_conv2d_6_bias: Y
Kassignvariableop_16_res_block_cnn_block_3_batch_normalization_3_moving_mean: ]
Oassignvariableop_17_res_block_cnn_block_3_batch_normalization_3_moving_variance: Y
Kassignvariableop_18_res_block_cnn_block_4_batch_normalization_4_moving_mean: ]
Oassignvariableop_19_res_block_cnn_block_4_batch_normalization_4_moving_variance: Y
Kassignvariableop_20_res_block_cnn_block_5_batch_normalization_5_moving_mean:@]
Oassignvariableop_21_res_block_cnn_block_5_batch_normalization_5_moving_variance:@V
;assignvariableop_22_res_block_1_cnn_block_6_conv2d_7_kernel:@H
9assignvariableop_23_res_block_1_cnn_block_6_conv2d_7_bias:	V
Gassignvariableop_24_res_block_1_cnn_block_6_batch_normalization_6_gamma:	U
Fassignvariableop_25_res_block_1_cnn_block_6_batch_normalization_6_beta:	W
;assignvariableop_26_res_block_1_cnn_block_7_conv2d_8_kernel:H
9assignvariableop_27_res_block_1_cnn_block_7_conv2d_8_bias:	V
Gassignvariableop_28_res_block_1_cnn_block_7_batch_normalization_7_gamma:	U
Fassignvariableop_29_res_block_1_cnn_block_7_batch_normalization_7_beta:	W
;assignvariableop_30_res_block_1_cnn_block_8_conv2d_9_kernel:H
9assignvariableop_31_res_block_1_cnn_block_8_conv2d_9_bias:	V
Gassignvariableop_32_res_block_1_cnn_block_8_batch_normalization_8_gamma:	U
Fassignvariableop_33_res_block_1_cnn_block_8_batch_normalization_8_beta:	K
0assignvariableop_34_res_block_1_conv2d_10_kernel:@=
.assignvariableop_35_res_block_1_conv2d_10_bias:	\
Massignvariableop_36_res_block_1_cnn_block_6_batch_normalization_6_moving_mean:	`
Qassignvariableop_37_res_block_1_cnn_block_6_batch_normalization_6_moving_variance:	\
Massignvariableop_38_res_block_1_cnn_block_7_batch_normalization_7_moving_mean:	`
Qassignvariableop_39_res_block_1_cnn_block_7_batch_normalization_7_moving_variance:	\
Massignvariableop_40_res_block_1_cnn_block_8_batch_normalization_8_moving_mean:	`
Qassignvariableop_41_res_block_1_cnn_block_8_batch_normalization_8_moving_variance:	'
assignvariableop_42_adam_iter:	 )
assignvariableop_43_adam_beta_1: )
assignvariableop_44_adam_beta_2: (
assignvariableop_45_adam_decay: 0
&assignvariableop_46_adam_learning_rate: %
assignvariableop_47_total_1: %
assignvariableop_48_count_1: #
assignvariableop_49_total: #
assignvariableop_50_count: <
)assignvariableop_51_adam_dense_2_kernel_m:	b
5
'assignvariableop_52_adam_dense_2_bias_m:
Z
@assignvariableop_53_adam_res_block_cnn_block_3_conv2d_3_kernel_m: L
>assignvariableop_54_adam_res_block_cnn_block_3_conv2d_3_bias_m: Z
Lassignvariableop_55_adam_res_block_cnn_block_3_batch_normalization_3_gamma_m: Y
Kassignvariableop_56_adam_res_block_cnn_block_3_batch_normalization_3_beta_m: Z
@assignvariableop_57_adam_res_block_cnn_block_4_conv2d_4_kernel_m:  L
>assignvariableop_58_adam_res_block_cnn_block_4_conv2d_4_bias_m: Z
Lassignvariableop_59_adam_res_block_cnn_block_4_batch_normalization_4_gamma_m: Y
Kassignvariableop_60_adam_res_block_cnn_block_4_batch_normalization_4_beta_m: Z
@assignvariableop_61_adam_res_block_cnn_block_5_conv2d_5_kernel_m: @L
>assignvariableop_62_adam_res_block_cnn_block_5_conv2d_5_bias_m:@Z
Lassignvariableop_63_adam_res_block_cnn_block_5_batch_normalization_5_gamma_m:@Y
Kassignvariableop_64_adam_res_block_cnn_block_5_batch_normalization_5_beta_m:@N
4assignvariableop_65_adam_res_block_conv2d_6_kernel_m: @
2assignvariableop_66_adam_res_block_conv2d_6_bias_m: ]
Bassignvariableop_67_adam_res_block_1_cnn_block_6_conv2d_7_kernel_m:@O
@assignvariableop_68_adam_res_block_1_cnn_block_6_conv2d_7_bias_m:	]
Nassignvariableop_69_adam_res_block_1_cnn_block_6_batch_normalization_6_gamma_m:	\
Massignvariableop_70_adam_res_block_1_cnn_block_6_batch_normalization_6_beta_m:	^
Bassignvariableop_71_adam_res_block_1_cnn_block_7_conv2d_8_kernel_m:O
@assignvariableop_72_adam_res_block_1_cnn_block_7_conv2d_8_bias_m:	]
Nassignvariableop_73_adam_res_block_1_cnn_block_7_batch_normalization_7_gamma_m:	\
Massignvariableop_74_adam_res_block_1_cnn_block_7_batch_normalization_7_beta_m:	^
Bassignvariableop_75_adam_res_block_1_cnn_block_8_conv2d_9_kernel_m:O
@assignvariableop_76_adam_res_block_1_cnn_block_8_conv2d_9_bias_m:	]
Nassignvariableop_77_adam_res_block_1_cnn_block_8_batch_normalization_8_gamma_m:	\
Massignvariableop_78_adam_res_block_1_cnn_block_8_batch_normalization_8_beta_m:	R
7assignvariableop_79_adam_res_block_1_conv2d_10_kernel_m:@D
5assignvariableop_80_adam_res_block_1_conv2d_10_bias_m:	<
)assignvariableop_81_adam_dense_2_kernel_v:	b
5
'assignvariableop_82_adam_dense_2_bias_v:
Z
@assignvariableop_83_adam_res_block_cnn_block_3_conv2d_3_kernel_v: L
>assignvariableop_84_adam_res_block_cnn_block_3_conv2d_3_bias_v: Z
Lassignvariableop_85_adam_res_block_cnn_block_3_batch_normalization_3_gamma_v: Y
Kassignvariableop_86_adam_res_block_cnn_block_3_batch_normalization_3_beta_v: Z
@assignvariableop_87_adam_res_block_cnn_block_4_conv2d_4_kernel_v:  L
>assignvariableop_88_adam_res_block_cnn_block_4_conv2d_4_bias_v: Z
Lassignvariableop_89_adam_res_block_cnn_block_4_batch_normalization_4_gamma_v: Y
Kassignvariableop_90_adam_res_block_cnn_block_4_batch_normalization_4_beta_v: Z
@assignvariableop_91_adam_res_block_cnn_block_5_conv2d_5_kernel_v: @L
>assignvariableop_92_adam_res_block_cnn_block_5_conv2d_5_bias_v:@Z
Lassignvariableop_93_adam_res_block_cnn_block_5_batch_normalization_5_gamma_v:@Y
Kassignvariableop_94_adam_res_block_cnn_block_5_batch_normalization_5_beta_v:@N
4assignvariableop_95_adam_res_block_conv2d_6_kernel_v: @
2assignvariableop_96_adam_res_block_conv2d_6_bias_v: ]
Bassignvariableop_97_adam_res_block_1_cnn_block_6_conv2d_7_kernel_v:@O
@assignvariableop_98_adam_res_block_1_cnn_block_6_conv2d_7_bias_v:	]
Nassignvariableop_99_adam_res_block_1_cnn_block_6_batch_normalization_6_gamma_v:	]
Nassignvariableop_100_adam_res_block_1_cnn_block_6_batch_normalization_6_beta_v:	_
Cassignvariableop_101_adam_res_block_1_cnn_block_7_conv2d_8_kernel_v:P
Aassignvariableop_102_adam_res_block_1_cnn_block_7_conv2d_8_bias_v:	^
Oassignvariableop_103_adam_res_block_1_cnn_block_7_batch_normalization_7_gamma_v:	]
Nassignvariableop_104_adam_res_block_1_cnn_block_7_batch_normalization_7_beta_v:	_
Cassignvariableop_105_adam_res_block_1_cnn_block_8_conv2d_9_kernel_v:P
Aassignvariableop_106_adam_res_block_1_cnn_block_8_conv2d_9_bias_v:	^
Oassignvariableop_107_adam_res_block_1_cnn_block_8_batch_normalization_8_gamma_v:	]
Nassignvariableop_108_adam_res_block_1_cnn_block_8_batch_normalization_8_beta_v:	S
8assignvariableop_109_adam_res_block_1_conv2d_10_kernel_v:@E
6assignvariableop_110_adam_res_block_1_conv2d_10_bias_v:	
identity_112¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_100¢AssignVariableOp_101¢AssignVariableOp_102¢AssignVariableOp_103¢AssignVariableOp_104¢AssignVariableOp_105¢AssignVariableOp_106¢AssignVariableOp_107¢AssignVariableOp_108¢AssignVariableOp_109¢AssignVariableOp_11¢AssignVariableOp_110¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93¢AssignVariableOp_94¢AssignVariableOp_95¢AssignVariableOp_96¢AssignVariableOp_97¢AssignVariableOp_98¢AssignVariableOp_99æ2
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:p*
dtype0*2
value2Bÿ1pB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÓ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:p*
dtype0*õ
valueëBèpB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ñ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ö
_output_shapesÃ
À::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*~
dtypest
r2p	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_2AssignVariableOp8assignvariableop_2_res_block_cnn_block_3_conv2d_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_3AssignVariableOp6assignvariableop_3_res_block_cnn_block_3_conv2d_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_4AssignVariableOpDassignvariableop_4_res_block_cnn_block_3_batch_normalization_3_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_5AssignVariableOpCassignvariableop_5_res_block_cnn_block_3_batch_normalization_3_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_6AssignVariableOp8assignvariableop_6_res_block_cnn_block_4_conv2d_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_7AssignVariableOp6assignvariableop_7_res_block_cnn_block_4_conv2d_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_8AssignVariableOpDassignvariableop_8_res_block_cnn_block_4_batch_normalization_4_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_9AssignVariableOpCassignvariableop_9_res_block_cnn_block_4_batch_normalization_4_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_10AssignVariableOp9assignvariableop_10_res_block_cnn_block_5_conv2d_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_11AssignVariableOp7assignvariableop_11_res_block_cnn_block_5_conv2d_5_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_12AssignVariableOpEassignvariableop_12_res_block_cnn_block_5_batch_normalization_5_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_13AssignVariableOpDassignvariableop_13_res_block_cnn_block_5_batch_normalization_5_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp-assignvariableop_14_res_block_conv2d_6_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp+assignvariableop_15_res_block_conv2d_6_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_16AssignVariableOpKassignvariableop_16_res_block_cnn_block_3_batch_normalization_3_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_17AssignVariableOpOassignvariableop_17_res_block_cnn_block_3_batch_normalization_3_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_18AssignVariableOpKassignvariableop_18_res_block_cnn_block_4_batch_normalization_4_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_19AssignVariableOpOassignvariableop_19_res_block_cnn_block_4_batch_normalization_4_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_20AssignVariableOpKassignvariableop_20_res_block_cnn_block_5_batch_normalization_5_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_21AssignVariableOpOassignvariableop_21_res_block_cnn_block_5_batch_normalization_5_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_22AssignVariableOp;assignvariableop_22_res_block_1_cnn_block_6_conv2d_7_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_23AssignVariableOp9assignvariableop_23_res_block_1_cnn_block_6_conv2d_7_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_24AssignVariableOpGassignvariableop_24_res_block_1_cnn_block_6_batch_normalization_6_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_25AssignVariableOpFassignvariableop_25_res_block_1_cnn_block_6_batch_normalization_6_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_26AssignVariableOp;assignvariableop_26_res_block_1_cnn_block_7_conv2d_8_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_27AssignVariableOp9assignvariableop_27_res_block_1_cnn_block_7_conv2d_8_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_28AssignVariableOpGassignvariableop_28_res_block_1_cnn_block_7_batch_normalization_7_gammaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_29AssignVariableOpFassignvariableop_29_res_block_1_cnn_block_7_batch_normalization_7_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_30AssignVariableOp;assignvariableop_30_res_block_1_cnn_block_8_conv2d_9_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_31AssignVariableOp9assignvariableop_31_res_block_1_cnn_block_8_conv2d_9_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_32AssignVariableOpGassignvariableop_32_res_block_1_cnn_block_8_batch_normalization_8_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_33AssignVariableOpFassignvariableop_33_res_block_1_cnn_block_8_batch_normalization_8_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_34AssignVariableOp0assignvariableop_34_res_block_1_conv2d_10_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp.assignvariableop_35_res_block_1_conv2d_10_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_36AssignVariableOpMassignvariableop_36_res_block_1_cnn_block_6_batch_normalization_6_moving_meanIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_37AssignVariableOpQassignvariableop_37_res_block_1_cnn_block_6_batch_normalization_6_moving_varianceIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_38AssignVariableOpMassignvariableop_38_res_block_1_cnn_block_7_batch_normalization_7_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_39AssignVariableOpQassignvariableop_39_res_block_1_cnn_block_7_batch_normalization_7_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_40AssignVariableOpMassignvariableop_40_res_block_1_cnn_block_8_batch_normalization_8_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_41AssignVariableOpQassignvariableop_41_res_block_1_cnn_block_8_batch_normalization_8_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_42AssignVariableOpassignvariableop_42_adam_iterIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOpassignvariableop_43_adam_beta_1Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOpassignvariableop_44_adam_beta_2Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOpassignvariableop_45_adam_decayIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp&assignvariableop_46_adam_learning_rateIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOpassignvariableop_47_total_1Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOpassignvariableop_48_count_1Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOpassignvariableop_49_totalIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOpassignvariableop_50_countIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_dense_2_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_dense_2_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_53AssignVariableOp@assignvariableop_53_adam_res_block_cnn_block_3_conv2d_3_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_54AssignVariableOp>assignvariableop_54_adam_res_block_cnn_block_3_conv2d_3_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_55AssignVariableOpLassignvariableop_55_adam_res_block_cnn_block_3_batch_normalization_3_gamma_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_56AssignVariableOpKassignvariableop_56_adam_res_block_cnn_block_3_batch_normalization_3_beta_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_57AssignVariableOp@assignvariableop_57_adam_res_block_cnn_block_4_conv2d_4_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_58AssignVariableOp>assignvariableop_58_adam_res_block_cnn_block_4_conv2d_4_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_59AssignVariableOpLassignvariableop_59_adam_res_block_cnn_block_4_batch_normalization_4_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_60AssignVariableOpKassignvariableop_60_adam_res_block_cnn_block_4_batch_normalization_4_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_61AssignVariableOp@assignvariableop_61_adam_res_block_cnn_block_5_conv2d_5_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_62AssignVariableOp>assignvariableop_62_adam_res_block_cnn_block_5_conv2d_5_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_63AssignVariableOpLassignvariableop_63_adam_res_block_cnn_block_5_batch_normalization_5_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_64AssignVariableOpKassignvariableop_64_adam_res_block_cnn_block_5_batch_normalization_5_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_65AssignVariableOp4assignvariableop_65_adam_res_block_conv2d_6_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_66AssignVariableOp2assignvariableop_66_adam_res_block_conv2d_6_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_67AssignVariableOpBassignvariableop_67_adam_res_block_1_cnn_block_6_conv2d_7_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_68AssignVariableOp@assignvariableop_68_adam_res_block_1_cnn_block_6_conv2d_7_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_69AssignVariableOpNassignvariableop_69_adam_res_block_1_cnn_block_6_batch_normalization_6_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_70AssignVariableOpMassignvariableop_70_adam_res_block_1_cnn_block_6_batch_normalization_6_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_71AssignVariableOpBassignvariableop_71_adam_res_block_1_cnn_block_7_conv2d_8_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_72AssignVariableOp@assignvariableop_72_adam_res_block_1_cnn_block_7_conv2d_8_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_73AssignVariableOpNassignvariableop_73_adam_res_block_1_cnn_block_7_batch_normalization_7_gamma_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_74AssignVariableOpMassignvariableop_74_adam_res_block_1_cnn_block_7_batch_normalization_7_beta_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_75AssignVariableOpBassignvariableop_75_adam_res_block_1_cnn_block_8_conv2d_9_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_76AssignVariableOp@assignvariableop_76_adam_res_block_1_cnn_block_8_conv2d_9_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_77AssignVariableOpNassignvariableop_77_adam_res_block_1_cnn_block_8_batch_normalization_8_gamma_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_78AssignVariableOpMassignvariableop_78_adam_res_block_1_cnn_block_8_batch_normalization_8_beta_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_79AssignVariableOp7assignvariableop_79_adam_res_block_1_conv2d_10_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_80AssignVariableOp5assignvariableop_80_adam_res_block_1_conv2d_10_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_81AssignVariableOp)assignvariableop_81_adam_dense_2_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_82AssignVariableOp'assignvariableop_82_adam_dense_2_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_83AssignVariableOp@assignvariableop_83_adam_res_block_cnn_block_3_conv2d_3_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_84AssignVariableOp>assignvariableop_84_adam_res_block_cnn_block_3_conv2d_3_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_85AssignVariableOpLassignvariableop_85_adam_res_block_cnn_block_3_batch_normalization_3_gamma_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_86AssignVariableOpKassignvariableop_86_adam_res_block_cnn_block_3_batch_normalization_3_beta_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_87AssignVariableOp@assignvariableop_87_adam_res_block_cnn_block_4_conv2d_4_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_88AssignVariableOp>assignvariableop_88_adam_res_block_cnn_block_4_conv2d_4_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_89AssignVariableOpLassignvariableop_89_adam_res_block_cnn_block_4_batch_normalization_4_gamma_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_90AssignVariableOpKassignvariableop_90_adam_res_block_cnn_block_4_batch_normalization_4_beta_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_91AssignVariableOp@assignvariableop_91_adam_res_block_cnn_block_5_conv2d_5_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_92AssignVariableOp>assignvariableop_92_adam_res_block_cnn_block_5_conv2d_5_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_93AssignVariableOpLassignvariableop_93_adam_res_block_cnn_block_5_batch_normalization_5_gamma_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_94AssignVariableOpKassignvariableop_94_adam_res_block_cnn_block_5_batch_normalization_5_beta_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_95AssignVariableOp4assignvariableop_95_adam_res_block_conv2d_6_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_96AssignVariableOp2assignvariableop_96_adam_res_block_conv2d_6_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_97AssignVariableOpBassignvariableop_97_adam_res_block_1_cnn_block_6_conv2d_7_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_98AssignVariableOp@assignvariableop_98_adam_res_block_1_cnn_block_6_conv2d_7_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_99AssignVariableOpNassignvariableop_99_adam_res_block_1_cnn_block_6_batch_normalization_6_gamma_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_100AssignVariableOpNassignvariableop_100_adam_res_block_1_cnn_block_6_batch_normalization_6_beta_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_101AssignVariableOpCassignvariableop_101_adam_res_block_1_cnn_block_7_conv2d_8_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_102AssignVariableOpAassignvariableop_102_adam_res_block_1_cnn_block_7_conv2d_8_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_103AssignVariableOpOassignvariableop_103_adam_res_block_1_cnn_block_7_batch_normalization_7_gamma_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_104AssignVariableOpNassignvariableop_104_adam_res_block_1_cnn_block_7_batch_normalization_7_beta_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_105AssignVariableOpCassignvariableop_105_adam_res_block_1_cnn_block_8_conv2d_9_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_106AssignVariableOpAassignvariableop_106_adam_res_block_1_cnn_block_8_conv2d_9_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_107AssignVariableOpOassignvariableop_107_adam_res_block_1_cnn_block_8_batch_normalization_8_gamma_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_108AssignVariableOpNassignvariableop_108_adam_res_block_1_cnn_block_8_batch_normalization_8_beta_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_109AssignVariableOp8assignvariableop_109_adam_res_block_1_conv2d_10_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_110AssignVariableOp6assignvariableop_110_adam_res_block_1_conv2d_10_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 å
Identity_111Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_112IdentityIdentity_111:output:0^NoOp_1*
T0*
_output_shapes
: Ñ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_112Identity_112:output:0*õ
_input_shapesã
à: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102*
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
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
	
Ô
5__inference_batch_normalization_6_layer_call_fn_10079

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
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
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7359
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
 
_user_specified_nameinputs"¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*²
serving_default
C
input_18
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ;
dense_20
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:²®
ò
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
ô
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
channels
cnn1
cnn2
cnn3
pooling
identity_mapping"
_tf_keras_layer
ô
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
!channels
"cnn1
#cnn2
$cnn3
%pooling
&identity_mapping"
_tf_keras_layer
¥
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
»
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias"
_tf_keras_layer
æ
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
D15
E16
F17
G18
H19
I20
J21
K22
L23
M24
N25
O26
P27
Q28
R29
S30
T31
U32
V33
W34
X35
Y36
Z37
[38
\39
340
441"
trackable_list_wrapper

50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
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
U26
V27
328
429"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Î
btrace_0
ctrace_1
dtrace_2
etrace_32ã
&__inference_model_1_layer_call_fn_7863
&__inference_model_1_layer_call_fn_8940
&__inference_model_1_layer_call_fn_9029
&__inference_model_1_layer_call_fn_8570À
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
 zbtrace_0zctrace_1zdtrace_2zetrace_3
º
ftrace_0
gtrace_1
htrace_2
itrace_32Ï
A__inference_model_1_layer_call_and_return_conditional_losses_9183
A__inference_model_1_layer_call_and_return_conditional_losses_9337
A__inference_model_1_layer_call_and_return_conditional_losses_8662
A__inference_model_1_layer_call_and_return_conditional_losses_8754À
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
 zftrace_0zgtrace_1zhtrace_2zitrace_3
ÊBÇ
__inference__wrapped_model_7102input_1"
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
«
jiter

kbeta_1

lbeta_2
	mdecay
nlearning_rate3mÃ4mÄ5mÅ6mÆ7mÇ8mÈ9mÉ:mÊ;mË<mÌ=mÍ>mÎ?mÏ@mÐAmÑBmÒImÓJmÔKmÕLmÖMm×NmØOmÙPmÚQmÛRmÜSmÝTmÞUmßVmà3vá4vâ5vã6vä7vå8væ9vç:vè;vé<vê=vë>vì?ví@vîAvïBvðIvñJvòKvóLvôMvõNvöOv÷PvøQvùRvúSvûTvüUvýVvþ"
	optimizer
,
oserving_default"
signature_map
¶
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
D15
E16
F17
G18
H19"
trackable_list_wrapper

50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13"
trackable_list_wrapper
 "
trackable_list_wrapper
­
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
È
utrace_0
vtrace_12
(__inference_res_block_layer_call_fn_9382
(__inference_res_block_layer_call_fn_9427º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
 zutrace_0zvtrace_1
þ
wtrace_0
xtrace_12Ç
C__inference_res_block_layer_call_and_return_conditional_losses_9502
C__inference_res_block_layer_call_and_return_conditional_losses_9577º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
 zwtrace_0zxtrace_1
 "
trackable_list_wrapper
¸
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses
conv
bn"
_tf_keras_layer
¿
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	conv
bn"
_tf_keras_layer
¿
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	conv
bn"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ä
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

Akernel
Bbias
!_jit_compiled_convolution_op"
_tf_keras_layer
¶
I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13
W14
X15
Y16
Z17
[18
\19"
trackable_list_wrapper

I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
Ð
£trace_0
¤trace_12
*__inference_res_block_1_layer_call_fn_9622
*__inference_res_block_1_layer_call_fn_9667º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
 z£trace_0z¤trace_1

¥trace_0
¦trace_12Ë
E__inference_res_block_1_layer_call_and_return_conditional_losses_9742
E__inference_res_block_1_layer_call_and_return_conditional_losses_9817º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
 z¥trace_0z¦trace_1
 "
trackable_list_wrapper
¿
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«__call__
+¬&call_and_return_all_conditional_losses
	­conv
®bn"
_tf_keras_layer
¿
¯	variables
°trainable_variables
±regularization_losses
²	keras_api
³__call__
+´&call_and_return_all_conditional_losses
	µconv
¶bn"
_tf_keras_layer
¿
·	variables
¸trainable_variables
¹regularization_losses
º	keras_api
»__call__
+¼&call_and_return_all_conditional_losses
	½conv
¾bn"
_tf_keras_layer
«
¿	variables
Àtrainable_variables
Áregularization_losses
Â	keras_api
Ã__call__
+Ä&call_and_return_all_conditional_losses"
_tf_keras_layer
ä
Å	variables
Ætrainable_variables
Çregularization_losses
È	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses

Ukernel
Vbias
!Ë_jit_compiled_convolution_op"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
î
Ñtrace_02Ï
(__inference_flatten_1_layer_call_fn_9822¢
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
 zÑtrace_0

Òtrace_02ê
C__inference_flatten_1_layer_call_and_return_conditional_losses_9828¢
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
 zÒtrace_0
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ónon_trainable_variables
Ôlayers
Õmetrics
 Ölayer_regularization_losses
×layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
ì
Øtrace_02Í
&__inference_dense_2_layer_call_fn_9837¢
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
 zØtrace_0

Ùtrace_02è
A__inference_dense_2_layer_call_and_return_conditional_losses_9847¢
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
 zÙtrace_0
!:	b
2dense_2/kernel
:
2dense_2/bias
?:= 2%res_block/cnn_block_3/conv2d_3/kernel
1:/ 2#res_block/cnn_block_3/conv2d_3/bias
?:= 21res_block/cnn_block_3/batch_normalization_3/gamma
>:< 20res_block/cnn_block_3/batch_normalization_3/beta
?:=  2%res_block/cnn_block_4/conv2d_4/kernel
1:/ 2#res_block/cnn_block_4/conv2d_4/bias
?:= 21res_block/cnn_block_4/batch_normalization_4/gamma
>:< 20res_block/cnn_block_4/batch_normalization_4/beta
?:= @2%res_block/cnn_block_5/conv2d_5/kernel
1:/@2#res_block/cnn_block_5/conv2d_5/bias
?:=@21res_block/cnn_block_5/batch_normalization_5/gamma
>:<@20res_block/cnn_block_5/batch_normalization_5/beta
3:1 2res_block/conv2d_6/kernel
%:# 2res_block/conv2d_6/bias
G:E  (27res_block/cnn_block_3/batch_normalization_3/moving_mean
K:I  (2;res_block/cnn_block_3/batch_normalization_3/moving_variance
G:E  (27res_block/cnn_block_4/batch_normalization_4/moving_mean
K:I  (2;res_block/cnn_block_4/batch_normalization_4/moving_variance
G:E@ (27res_block/cnn_block_5/batch_normalization_5/moving_mean
K:I@ (2;res_block/cnn_block_5/batch_normalization_5/moving_variance
B:@@2'res_block_1/cnn_block_6/conv2d_7/kernel
4:22%res_block_1/cnn_block_6/conv2d_7/bias
B:@23res_block_1/cnn_block_6/batch_normalization_6/gamma
A:?22res_block_1/cnn_block_6/batch_normalization_6/beta
C:A2'res_block_1/cnn_block_7/conv2d_8/kernel
4:22%res_block_1/cnn_block_7/conv2d_8/bias
B:@23res_block_1/cnn_block_7/batch_normalization_7/gamma
A:?22res_block_1/cnn_block_7/batch_normalization_7/beta
C:A2'res_block_1/cnn_block_8/conv2d_9/kernel
4:22%res_block_1/cnn_block_8/conv2d_9/bias
B:@23res_block_1/cnn_block_8/batch_normalization_8/gamma
A:?22res_block_1/cnn_block_8/batch_normalization_8/beta
7:5@2res_block_1/conv2d_10/kernel
):'2res_block_1/conv2d_10/bias
J:H (29res_block_1/cnn_block_6/batch_normalization_6/moving_mean
N:L (2=res_block_1/cnn_block_6/batch_normalization_6/moving_variance
J:H (29res_block_1/cnn_block_7/batch_normalization_7/moving_mean
N:L (2=res_block_1/cnn_block_7/batch_normalization_7/moving_variance
J:H (29res_block_1/cnn_block_8/batch_normalization_8/moving_mean
N:L (2=res_block_1/cnn_block_8/batch_normalization_8/moving_variance
v
C0
D1
E2
F3
G4
H5
W6
X7
Y8
Z9
[10
\11"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
0
Ú0
Û1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ùBö
&__inference_model_1_layer_call_fn_7863input_1"À
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
øBõ
&__inference_model_1_layer_call_fn_8940inputs"À
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
øBõ
&__inference_model_1_layer_call_fn_9029inputs"À
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
ùBö
&__inference_model_1_layer_call_fn_8570input_1"À
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
B
A__inference_model_1_layer_call_and_return_conditional_losses_9183inputs"À
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
B
A__inference_model_1_layer_call_and_return_conditional_losses_9337inputs"À
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
B
A__inference_model_1_layer_call_and_return_conditional_losses_8662input_1"À
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
B
A__inference_model_1_layer_call_and_return_conditional_losses_8754input_1"À
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÉBÆ
"__inference_signature_wrapper_8851input_1"
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
J
C0
D1
E2
F3
G4
H5"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
úB÷
(__inference_res_block_layer_call_fn_9382input_tensor"º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
úB÷
(__inference_res_block_layer_call_fn_9427input_tensor"º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
B
C__inference_res_block_layer_call_and_return_conditional_losses_9502input_tensor"º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
B
C__inference_res_block_layer_call_and_return_conditional_losses_9577input_tensor"º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
J
50
61
72
83
C4
D5"
trackable_list_wrapper
<
50
61
72
83"
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
À2½º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
À2½º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
ä
á	variables
âtrainable_variables
ãregularization_losses
ä	keras_api
å__call__
+æ&call_and_return_all_conditional_losses

5kernel
6bias
!ç_jit_compiled_convolution_op"
_tf_keras_layer
ñ
è	variables
étrainable_variables
êregularization_losses
ë	keras_api
ì__call__
+í&call_and_return_all_conditional_losses
	îaxis
	7gamma
8beta
Cmoving_mean
Dmoving_variance"
_tf_keras_layer
J
90
:1
;2
<3
E4
F5"
trackable_list_wrapper
<
90
:1
;2
<3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
À2½º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
À2½º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
ä
ô	variables
õtrainable_variables
öregularization_losses
÷	keras_api
ø__call__
+ù&call_and_return_all_conditional_losses

9kernel
:bias
!ú_jit_compiled_convolution_op"
_tf_keras_layer
ñ
û	variables
ütrainable_variables
ýregularization_losses
þ	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses
	axis
	;gamma
<beta
Emoving_mean
Fmoving_variance"
_tf_keras_layer
J
=0
>1
?2
@3
G4
H5"
trackable_list_wrapper
<
=0
>1
?2
@3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
À2½º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
À2½º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
ä
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

=kernel
>bias
!_jit_compiled_convolution_op"
_tf_keras_layer
ñ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis
	?gamma
@beta
Gmoving_mean
Hmoving_variance"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ò
trace_02Ó
,__inference_max_pooling2d_layer_call_fn_9852¢
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
 ztrace_0

trace_02î
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_9857¢
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
 ztrace_0
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
 0
J
W0
X1
Y2
Z3
[4
\5"
trackable_list_wrapper
C
"0
#1
$2
%3
&4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
üBù
*__inference_res_block_1_layer_call_fn_9622input_tensor"º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
üBù
*__inference_res_block_1_layer_call_fn_9667input_tensor"º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
B
E__inference_res_block_1_layer_call_and_return_conditional_losses_9742input_tensor"º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
B
E__inference_res_block_1_layer_call_and_return_conditional_losses_9817input_tensor"º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
J
I0
J1
K2
L3
W4
X5"
trackable_list_wrapper
<
I0
J1
K2
L3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
À2½º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
À2½º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
ä
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
ª__call__
+«&call_and_return_all_conditional_losses

Ikernel
Jbias
!¬_jit_compiled_convolution_op"
_tf_keras_layer
ñ
­	variables
®trainable_variables
¯regularization_losses
°	keras_api
±__call__
+²&call_and_return_all_conditional_losses
	³axis
	Kgamma
Lbeta
Wmoving_mean
Xmoving_variance"
_tf_keras_layer
J
M0
N1
O2
P3
Y4
Z5"
trackable_list_wrapper
<
M0
N1
O2
P3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
¯	variables
°trainable_variables
±regularization_losses
³__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
À2½º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
À2½º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
ä
¹	variables
ºtrainable_variables
»regularization_losses
¼	keras_api
½__call__
+¾&call_and_return_all_conditional_losses

Mkernel
Nbias
!¿_jit_compiled_convolution_op"
_tf_keras_layer
ñ
À	variables
Átrainable_variables
Âregularization_losses
Ã	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses
	Æaxis
	Ogamma
Pbeta
Ymoving_mean
Zmoving_variance"
_tf_keras_layer
J
Q0
R1
S2
T3
[4
\5"
trackable_list_wrapper
<
Q0
R1
S2
T3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
·	variables
¸trainable_variables
¹regularization_losses
»__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses"
_generic_user_object
À2½º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
À2½º
±²­
FullArgSpec/
args'$
jself
jinput_tensor

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
ä
Ì	variables
Ítrainable_variables
Îregularization_losses
Ï	keras_api
Ð__call__
+Ñ&call_and_return_all_conditional_losses

Qkernel
Rbias
!Ò_jit_compiled_convolution_op"
_tf_keras_layer
ñ
Ó	variables
Ôtrainable_variables
Õregularization_losses
Ö	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses
	Ùaxis
	Sgamma
Tbeta
[moving_mean
\moving_variance"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Únon_trainable_variables
Ûlayers
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
¿	variables
Àtrainable_variables
Áregularization_losses
Ã__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses"
_generic_user_object
ô
ßtrace_02Õ
.__inference_max_pooling2d_1_layer_call_fn_9862¢
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
 zßtrace_0

àtrace_02ð
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_9867¢
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
 zàtrace_0
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
Å	variables
Ætrainable_variables
Çregularization_losses
É__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
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
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
 0
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
ÜBÙ
(__inference_flatten_1_layer_call_fn_9822inputs"¢
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
÷Bô
C__inference_flatten_1_layer_call_and_return_conditional_losses_9828inputs"¢
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÚB×
&__inference_dense_2_layer_call_fn_9837inputs"¢
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
õBò
A__inference_dense_2_layer_call_and_return_conditional_losses_9847inputs"¢
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
R
æ	variables
ç	keras_api

ètotal

écount"
_tf_keras_metric
c
ê	variables
ë	keras_api

ìtotal

ícount
î
_fn_kwargs"
_tf_keras_metric
.
C0
D1"
trackable_list_wrapper
/
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
á	variables
âtrainable_variables
ãregularization_losses
å__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses"
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
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
 0
<
70
81
C2
D3"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
è	variables
étrainable_variables
êregularization_losses
ì__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses"
_generic_user_object
Þ
ùtrace_0
útrace_12£
4__inference_batch_normalization_3_layer_call_fn_9880
4__inference_batch_normalization_3_layer_call_fn_9893´
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
 zùtrace_0zútrace_1

ûtrace_0
ütrace_12Ù
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9911
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9929´
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
 zûtrace_0zütrace_1
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ýnon_trainable_variables
þlayers
ÿmetrics
 layer_regularization_losses
layer_metrics
ô	variables
õtrainable_variables
öregularization_losses
ø__call__
+ù&call_and_return_all_conditional_losses
'ù"call_and_return_conditional_losses"
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
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
 0
<
;0
<1
E2
F3"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
û	variables
ütrainable_variables
ýregularization_losses
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Þ
trace_0
trace_12£
4__inference_batch_normalization_4_layer_call_fn_9942
4__inference_batch_normalization_4_layer_call_fn_9955´
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
 ztrace_0ztrace_1

trace_0
trace_12Ù
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9973
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9991´
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
 ztrace_0ztrace_1
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
 0
<
?0
@1
G2
H3"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
à
trace_0
trace_12¥
5__inference_batch_normalization_5_layer_call_fn_10004
5__inference_batch_normalization_5_layer_call_fn_10017´
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
 ztrace_0ztrace_1

trace_0
trace_12Û
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10035
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10053´
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
 ztrace_0ztrace_1
 "
trackable_list_wrapper
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
àBÝ
,__inference_max_pooling2d_layer_call_fn_9852inputs"¢
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
ûBø
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_9857inputs"¢
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
W0
X1"
trackable_list_wrapper
0
­0
®1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¦	variables
§trainable_variables
¨regularization_losses
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
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
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
 0
<
K0
L1
W2
X3"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
­	variables
®trainable_variables
¯regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
à
£trace_0
¤trace_12¥
5__inference_batch_normalization_6_layer_call_fn_10066
5__inference_batch_normalization_6_layer_call_fn_10079´
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
 z£trace_0z¤trace_1

¥trace_0
¦trace_12Û
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10097
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10115´
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
 z¥trace_0z¦trace_1
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
0
µ0
¶1"
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
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
¹	variables
ºtrainable_variables
»regularization_losses
½__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses"
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
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
 0
<
O0
P1
Y2
Z3"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
À	variables
Átrainable_variables
Âregularization_losses
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
à
±trace_0
²trace_12¥
5__inference_batch_normalization_7_layer_call_fn_10128
5__inference_batch_normalization_7_layer_call_fn_10141´
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
 z±trace_0z²trace_1

³trace_0
´trace_12Û
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10159
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10177´
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
 z³trace_0z´trace_1
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
0
½0
¾1"
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
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
µnon_trainable_variables
¶layers
·metrics
 ¸layer_regularization_losses
¹layer_metrics
Ì	variables
Ítrainable_variables
Îregularization_losses
Ð__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses"
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
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
 0
<
S0
T1
[2
\3"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ºnon_trainable_variables
»layers
¼metrics
 ½layer_regularization_losses
¾layer_metrics
Ó	variables
Ôtrainable_variables
Õregularization_losses
×__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
à
¿trace_0
Àtrace_12¥
5__inference_batch_normalization_8_layer_call_fn_10190
5__inference_batch_normalization_8_layer_call_fn_10203´
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
 z¿trace_0zÀtrace_1

Átrace_0
Âtrace_12Û
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10221
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10239´
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
 zÁtrace_0zÂtrace_1
 "
trackable_list_wrapper
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
âBß
.__inference_max_pooling2d_1_layer_call_fn_9862inputs"¢
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
ýBú
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_9867inputs"¢
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
è0
é1"
trackable_list_wrapper
.
æ	variables"
_generic_user_object
:  (2total
:  (2count
0
ì0
í1"
trackable_list_wrapper
.
ê	variables"
_generic_user_object
:  (2total
:  (2count
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
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
úB÷
4__inference_batch_normalization_3_layer_call_fn_9880inputs"´
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
úB÷
4__inference_batch_normalization_3_layer_call_fn_9893inputs"´
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
B
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9911inputs"´
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
B
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9929inputs"´
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
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
úB÷
4__inference_batch_normalization_4_layer_call_fn_9942inputs"´
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
úB÷
4__inference_batch_normalization_4_layer_call_fn_9955inputs"´
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
B
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9973inputs"´
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
B
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9991inputs"´
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
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ûBø
5__inference_batch_normalization_5_layer_call_fn_10004inputs"´
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
ûBø
5__inference_batch_normalization_5_layer_call_fn_10017inputs"´
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
B
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10035inputs"´
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
B
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10053inputs"´
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
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ûBø
5__inference_batch_normalization_6_layer_call_fn_10066inputs"´
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
ûBø
5__inference_batch_normalization_6_layer_call_fn_10079inputs"´
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
B
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10097inputs"´
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
B
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10115inputs"´
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
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ûBø
5__inference_batch_normalization_7_layer_call_fn_10128inputs"´
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
ûBø
5__inference_batch_normalization_7_layer_call_fn_10141inputs"´
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
B
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10159inputs"´
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
B
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10177inputs"´
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
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ûBø
5__inference_batch_normalization_8_layer_call_fn_10190inputs"´
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
ûBø
5__inference_batch_normalization_8_layer_call_fn_10203inputs"´
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
B
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10221inputs"´
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
B
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10239inputs"´
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
&:$	b
2Adam/dense_2/kernel/m
:
2Adam/dense_2/bias/m
D:B 2,Adam/res_block/cnn_block_3/conv2d_3/kernel/m
6:4 2*Adam/res_block/cnn_block_3/conv2d_3/bias/m
D:B 28Adam/res_block/cnn_block_3/batch_normalization_3/gamma/m
C:A 27Adam/res_block/cnn_block_3/batch_normalization_3/beta/m
D:B  2,Adam/res_block/cnn_block_4/conv2d_4/kernel/m
6:4 2*Adam/res_block/cnn_block_4/conv2d_4/bias/m
D:B 28Adam/res_block/cnn_block_4/batch_normalization_4/gamma/m
C:A 27Adam/res_block/cnn_block_4/batch_normalization_4/beta/m
D:B @2,Adam/res_block/cnn_block_5/conv2d_5/kernel/m
6:4@2*Adam/res_block/cnn_block_5/conv2d_5/bias/m
D:B@28Adam/res_block/cnn_block_5/batch_normalization_5/gamma/m
C:A@27Adam/res_block/cnn_block_5/batch_normalization_5/beta/m
8:6 2 Adam/res_block/conv2d_6/kernel/m
*:( 2Adam/res_block/conv2d_6/bias/m
G:E@2.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/m
9:72,Adam/res_block_1/cnn_block_6/conv2d_7/bias/m
G:E2:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/m
F:D29Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/m
H:F2.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/m
9:72,Adam/res_block_1/cnn_block_7/conv2d_8/bias/m
G:E2:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/m
F:D29Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/m
H:F2.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/m
9:72,Adam/res_block_1/cnn_block_8/conv2d_9/bias/m
G:E2:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/m
F:D29Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/m
<::@2#Adam/res_block_1/conv2d_10/kernel/m
.:,2!Adam/res_block_1/conv2d_10/bias/m
&:$	b
2Adam/dense_2/kernel/v
:
2Adam/dense_2/bias/v
D:B 2,Adam/res_block/cnn_block_3/conv2d_3/kernel/v
6:4 2*Adam/res_block/cnn_block_3/conv2d_3/bias/v
D:B 28Adam/res_block/cnn_block_3/batch_normalization_3/gamma/v
C:A 27Adam/res_block/cnn_block_3/batch_normalization_3/beta/v
D:B  2,Adam/res_block/cnn_block_4/conv2d_4/kernel/v
6:4 2*Adam/res_block/cnn_block_4/conv2d_4/bias/v
D:B 28Adam/res_block/cnn_block_4/batch_normalization_4/gamma/v
C:A 27Adam/res_block/cnn_block_4/batch_normalization_4/beta/v
D:B @2,Adam/res_block/cnn_block_5/conv2d_5/kernel/v
6:4@2*Adam/res_block/cnn_block_5/conv2d_5/bias/v
D:B@28Adam/res_block/cnn_block_5/batch_normalization_5/gamma/v
C:A@27Adam/res_block/cnn_block_5/batch_normalization_5/beta/v
8:6 2 Adam/res_block/conv2d_6/kernel/v
*:( 2Adam/res_block/conv2d_6/bias/v
G:E@2.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/v
9:72,Adam/res_block_1/cnn_block_6/conv2d_7/bias/v
G:E2:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/v
F:D29Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/v
H:F2.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/v
9:72,Adam/res_block_1/cnn_block_7/conv2d_8/bias/v
G:E2:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/v
F:D29Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/v
H:F2.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/v
9:72,Adam/res_block_1/cnn_block_8/conv2d_9/bias/v
G:E2:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/v
F:D29Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/v
<::@2#Adam/res_block_1/conv2d_10/kernel/v
.:,2!Adam/res_block_1/conv2d_10/bias/v½
__inference__wrapped_model_7102*5678CD9:;<EFAB=>?@GHIJKLWXMNOPYZUVQRST[\348¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_2!
dense_2ÿÿÿÿÿÿÿÿÿ
ê
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_991178CDM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ê
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_992978CDM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Â
4__inference_batch_normalization_3_layer_call_fn_988078CDM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Â
4__inference_batch_normalization_3_layer_call_fn_989378CDM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ê
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9973;<EFM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ê
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9991;<EFM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Â
4__inference_batch_normalization_4_layer_call_fn_9942;<EFM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Â
4__inference_batch_normalization_4_layer_call_fn_9955;<EFM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ë
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10035?@GHM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ë
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10053?@GHM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ã
5__inference_batch_normalization_5_layer_call_fn_10004?@GHM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ã
5__inference_batch_normalization_5_layer_call_fn_10017?@GHM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@í
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10097KLWXN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 í
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10115KLWXN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
5__inference_batch_normalization_6_layer_call_fn_10066KLWXN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ
5__inference_batch_normalization_6_layer_call_fn_10079KLWXN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10159OPYZN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 í
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10177OPYZN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
5__inference_batch_normalization_7_layer_call_fn_10128OPYZN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ
5__inference_batch_normalization_7_layer_call_fn_10141OPYZN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10221ST[\N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 í
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10239ST[\N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
5__inference_batch_normalization_8_layer_call_fn_10190ST[\N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ
5__inference_batch_normalization_8_layer_call_fn_10203ST[\N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
A__inference_dense_2_layer_call_and_return_conditional_losses_9847]340¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿb
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 z
&__inference_dense_2_layer_call_fn_9837P340¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿb
ª "ÿÿÿÿÿÿÿÿÿ
©
C__inference_flatten_1_layer_call_and_return_conditional_losses_9828b8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿb
 
(__inference_flatten_1_layer_call_fn_9822U8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿbì
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_9867R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_max_pooling2d_1_layer_call_fn_9862R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿê
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_9857R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
,__inference_max_pooling2d_layer_call_fn_9852R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÛ
A__inference_model_1_layer_call_and_return_conditional_losses_8662*5678CD9:;<EFAB=>?@GHIJKLWXMNOPYZUVQRST[\34@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Û
A__inference_model_1_layer_call_and_return_conditional_losses_8754*5678CD9:;<EFAB=>?@GHIJKLWXMNOPYZUVQRST[\34@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ú
A__inference_model_1_layer_call_and_return_conditional_losses_9183*5678CD9:;<EFAB=>?@GHIJKLWXMNOPYZUVQRST[\34?¢<
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
A__inference_model_1_layer_call_and_return_conditional_losses_9337*5678CD9:;<EFAB=>?@GHIJKLWXMNOPYZUVQRST[\34?¢<
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
 ³
&__inference_model_1_layer_call_fn_7863*5678CD9:;<EFAB=>?@GHIJKLWXMNOPYZUVQRST[\34@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
³
&__inference_model_1_layer_call_fn_8570*5678CD9:;<EFAB=>?@GHIJKLWXMNOPYZUVQRST[\34@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
²
&__inference_model_1_layer_call_fn_8940*5678CD9:;<EFAB=>?@GHIJKLWXMNOPYZUVQRST[\34?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
²
&__inference_model_1_layer_call_fn_9029*5678CD9:;<EFAB=>?@GHIJKLWXMNOPYZUVQRST[\34?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
Ó
E__inference_res_block_1_layer_call_and_return_conditional_losses_9742IJKLWXMNOPYZUVQRST[\A¢>
7¢4
.+
input_tensorÿÿÿÿÿÿÿÿÿ@
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ó
E__inference_res_block_1_layer_call_and_return_conditional_losses_9817IJKLWXMNOPYZUVQRST[\A¢>
7¢4
.+
input_tensorÿÿÿÿÿÿÿÿÿ@
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ª
*__inference_res_block_1_layer_call_fn_9622|IJKLWXMNOPYZUVQRST[\A¢>
7¢4
.+
input_tensorÿÿÿÿÿÿÿÿÿ@
p 
ª "!ÿÿÿÿÿÿÿÿÿª
*__inference_res_block_1_layer_call_fn_9667|IJKLWXMNOPYZUVQRST[\A¢>
7¢4
.+
input_tensorÿÿÿÿÿÿÿÿÿ@
p
ª "!ÿÿÿÿÿÿÿÿÿÐ
C__inference_res_block_layer_call_and_return_conditional_losses_95025678CD9:;<EFAB=>?@GHA¢>
7¢4
.+
input_tensorÿÿÿÿÿÿÿÿÿ
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Ð
C__inference_res_block_layer_call_and_return_conditional_losses_95775678CD9:;<EFAB=>?@GHA¢>
7¢4
.+
input_tensorÿÿÿÿÿÿÿÿÿ
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 §
(__inference_res_block_layer_call_fn_9382{5678CD9:;<EFAB=>?@GHA¢>
7¢4
.+
input_tensorÿÿÿÿÿÿÿÿÿ
p 
ª " ÿÿÿÿÿÿÿÿÿ@§
(__inference_res_block_layer_call_fn_9427{5678CD9:;<EFAB=>?@GHA¢>
7¢4
.+
input_tensorÿÿÿÿÿÿÿÿÿ
p
ª " ÿÿÿÿÿÿÿÿÿ@Ë
"__inference_signature_wrapper_8851¤*5678CD9:;<EFAB=>?@GHIJKLWXMNOPYZUVQRST[\34C¢@
¢ 
9ª6
4
input_1)&
input_1ÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_2!
dense_2ÿÿÿÿÿÿÿÿÿ
