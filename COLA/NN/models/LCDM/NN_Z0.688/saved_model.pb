��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
0
Sigmoid
x"T
y"T"
Ttype:

2
�
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
executor_typestring ��
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758١
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
~
Adam/v/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_3/bias
w
'Adam/v/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_3/bias
w
'Adam/m/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameAdam/v/dense_3/kernel
�
)Adam/v/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameAdam/m/dense_3/kernel
�
)Adam/m/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/kernel*
_output_shapes
:	�*
dtype0
�
&Adam/v/custom_activation_layer_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&Adam/v/custom_activation_layer_2/gamma
�
:Adam/v/custom_activation_layer_2/gamma/Read/ReadVariableOpReadVariableOp&Adam/v/custom_activation_layer_2/gamma*
_output_shapes	
:�*
dtype0
�
&Adam/m/custom_activation_layer_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&Adam/m/custom_activation_layer_2/gamma
�
:Adam/m/custom_activation_layer_2/gamma/Read/ReadVariableOpReadVariableOp&Adam/m/custom_activation_layer_2/gamma*
_output_shapes	
:�*
dtype0
�
%Adam/v/custom_activation_layer_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%Adam/v/custom_activation_layer_2/beta
�
9Adam/v/custom_activation_layer_2/beta/Read/ReadVariableOpReadVariableOp%Adam/v/custom_activation_layer_2/beta*
_output_shapes	
:�*
dtype0
�
%Adam/m/custom_activation_layer_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%Adam/m/custom_activation_layer_2/beta
�
9Adam/m/custom_activation_layer_2/beta/Read/ReadVariableOpReadVariableOp%Adam/m/custom_activation_layer_2/beta*
_output_shapes	
:�*
dtype0

Adam/v/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/v/dense_2/bias
x
'Adam/v/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/bias*
_output_shapes	
:�*
dtype0

Adam/m/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/m/dense_2/bias
x
'Adam/m/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/v/dense_2/kernel
�
)Adam/v/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/m/dense_2/kernel
�
)Adam/m/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/kernel* 
_output_shapes
:
��*
dtype0
�
&Adam/v/custom_activation_layer_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&Adam/v/custom_activation_layer_1/gamma
�
:Adam/v/custom_activation_layer_1/gamma/Read/ReadVariableOpReadVariableOp&Adam/v/custom_activation_layer_1/gamma*
_output_shapes	
:�*
dtype0
�
&Adam/m/custom_activation_layer_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&Adam/m/custom_activation_layer_1/gamma
�
:Adam/m/custom_activation_layer_1/gamma/Read/ReadVariableOpReadVariableOp&Adam/m/custom_activation_layer_1/gamma*
_output_shapes	
:�*
dtype0
�
%Adam/v/custom_activation_layer_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%Adam/v/custom_activation_layer_1/beta
�
9Adam/v/custom_activation_layer_1/beta/Read/ReadVariableOpReadVariableOp%Adam/v/custom_activation_layer_1/beta*
_output_shapes	
:�*
dtype0
�
%Adam/m/custom_activation_layer_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%Adam/m/custom_activation_layer_1/beta
�
9Adam/m/custom_activation_layer_1/beta/Read/ReadVariableOpReadVariableOp%Adam/m/custom_activation_layer_1/beta*
_output_shapes	
:�*
dtype0

Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/v/dense_1/bias
x
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes	
:�*
dtype0

Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/m/dense_1/bias
x
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/v/dense_1/kernel
�
)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/m/dense_1/kernel
�
)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel* 
_output_shapes
:
��*
dtype0
�
$Adam/v/custom_activation_layer/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$Adam/v/custom_activation_layer/gamma
�
8Adam/v/custom_activation_layer/gamma/Read/ReadVariableOpReadVariableOp$Adam/v/custom_activation_layer/gamma*
_output_shapes	
:�*
dtype0
�
$Adam/m/custom_activation_layer/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$Adam/m/custom_activation_layer/gamma
�
8Adam/m/custom_activation_layer/gamma/Read/ReadVariableOpReadVariableOp$Adam/m/custom_activation_layer/gamma*
_output_shapes	
:�*
dtype0
�
#Adam/v/custom_activation_layer/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/v/custom_activation_layer/beta
�
7Adam/v/custom_activation_layer/beta/Read/ReadVariableOpReadVariableOp#Adam/v/custom_activation_layer/beta*
_output_shapes	
:�*
dtype0
�
#Adam/m/custom_activation_layer/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/m/custom_activation_layer/beta
�
7Adam/m/custom_activation_layer/beta/Read/ReadVariableOpReadVariableOp#Adam/m/custom_activation_layer/beta*
_output_shapes	
:�*
dtype0
{
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/v/dense/bias
t
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes	
:�*
dtype0
{
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/m/dense/bias
t
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_nameAdam/v/dense/kernel
|
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_nameAdam/m/dense/kernel
|
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel*
_output_shapes
:	�*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	�*
dtype0
�
custom_activation_layer_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!custom_activation_layer_2/gamma
�
3custom_activation_layer_2/gamma/Read/ReadVariableOpReadVariableOpcustom_activation_layer_2/gamma*
_output_shapes	
:�*
dtype0
�
custom_activation_layer_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*/
shared_name custom_activation_layer_2/beta
�
2custom_activation_layer_2/beta/Read/ReadVariableOpReadVariableOpcustom_activation_layer_2/beta*
_output_shapes	
:�*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:�*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
��*
dtype0
�
custom_activation_layer_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!custom_activation_layer_1/gamma
�
3custom_activation_layer_1/gamma/Read/ReadVariableOpReadVariableOpcustom_activation_layer_1/gamma*
_output_shapes	
:�*
dtype0
�
custom_activation_layer_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*/
shared_name custom_activation_layer_1/beta
�
2custom_activation_layer_1/beta/Read/ReadVariableOpReadVariableOpcustom_activation_layer_1/beta*
_output_shapes	
:�*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
��*
dtype0
�
custom_activation_layer/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_namecustom_activation_layer/gamma
�
1custom_activation_layer/gamma/Read/ReadVariableOpReadVariableOpcustom_activation_layer/gamma*
_output_shapes	
:�*
dtype0
�
custom_activation_layer/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namecustom_activation_layer/beta
�
0custom_activation_layer/beta/Read/ReadVariableOpReadVariableOpcustom_activation_layer/beta*
_output_shapes	
:�*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biascustom_activation_layer/betacustom_activation_layer/gammadense_1/kerneldense_1/biascustom_activation_layer_1/betacustom_activation_layer_1/gammadense_2/kerneldense_2/biascustom_activation_layer_2/betacustom_activation_layer_2/gammadense_3/kerneldense_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_135231

NoOpNoOp
�S
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�S
value�RB�R B�R
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 beta
	!gamma*
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias*
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0beta
	1gamma*
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias*
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@beta
	Agamma*
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

Hkernel
Ibias*
j
0
1
 2
!3
(4
)5
06
17
88
99
@10
A11
H12
I13*
j
0
1
 2
!3
(4
)5
06
17
88
99
@10
A11
H12
I13*
* 
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_3* 
6
Strace_0
Ttrace_1
Utrace_2
Vtrace_3* 
* 
�
W
_variables
X_iterations
Y_learning_rate
Z_index_dict
[
_momentums
\_velocities
]_update_step_xla*

^serving_default* 

0
1*

0
1*
* 
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

dtrace_0* 

etrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

 0
!1*

 0
!1*
* 
�
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

ktrace_0* 

ltrace_0* 
jd
VARIABLE_VALUEcustom_activation_layer/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEcustom_activation_layer/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*

(0
)1*

(0
)1*
* 
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

rtrace_0* 

strace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

00
11*

00
11*
* 
�
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

ytrace_0* 

ztrace_0* 
lf
VARIABLE_VALUEcustom_activation_layer_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEcustom_activation_layer_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*

80
91*

80
91*
* 
�
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

@0
A1*

@0
A1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
lf
VARIABLE_VALUEcustom_activation_layer_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEcustom_activation_layer_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*

H0
I1*

H0
I1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
0
1
2
3
4
5
6
7*

�0*
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
�
X0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
x
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13*
x
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13*
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
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
^X
VARIABLE_VALUEAdam/m/dense/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/dense/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/dense/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE#Adam/m/custom_activation_layer/beta1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE#Adam/v/custom_activation_layer/beta1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/m/custom_activation_layer/gamma1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/v/custom_activation_layer/gamma1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_1/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_1/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_1/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_1/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/m/custom_activation_layer_1/beta2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/v/custom_activation_layer_1/beta2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/m/custom_activation_layer_1/gamma2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/v/custom_activation_layer_1/gamma2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_2/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_2/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_2/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_2/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/m/custom_activation_layer_2/beta2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/v/custom_activation_layer_2/beta2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/m/custom_activation_layer_2/gamma2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/v/custom_activation_layer_2/gamma2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_3/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_3/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_3/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_3/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense/kernel
dense/biascustom_activation_layer/betacustom_activation_layer/gammadense_1/kerneldense_1/biascustom_activation_layer_1/betacustom_activation_layer_1/gammadense_2/kerneldense_2/biascustom_activation_layer_2/betacustom_activation_layer_2/gammadense_3/kerneldense_3/bias	iterationlearning_rateAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/bias#Adam/m/custom_activation_layer/beta#Adam/v/custom_activation_layer/beta$Adam/m/custom_activation_layer/gamma$Adam/v/custom_activation_layer/gammaAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/bias%Adam/m/custom_activation_layer_1/beta%Adam/v/custom_activation_layer_1/beta&Adam/m/custom_activation_layer_1/gamma&Adam/v/custom_activation_layer_1/gammaAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/bias%Adam/m/custom_activation_layer_2/beta%Adam/v/custom_activation_layer_2/beta&Adam/m/custom_activation_layer_2/gamma&Adam/v/custom_activation_layer_2/gammaAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biastotalcountConst*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_135875
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biascustom_activation_layer/betacustom_activation_layer/gammadense_1/kerneldense_1/biascustom_activation_layer_1/betacustom_activation_layer_1/gammadense_2/kerneldense_2/biascustom_activation_layer_2/betacustom_activation_layer_2/gammadense_3/kerneldense_3/bias	iterationlearning_rateAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/bias#Adam/m/custom_activation_layer/beta#Adam/v/custom_activation_layer/beta$Adam/m/custom_activation_layer/gamma$Adam/v/custom_activation_layer/gammaAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/bias%Adam/m/custom_activation_layer_1/beta%Adam/v/custom_activation_layer_1/beta&Adam/m/custom_activation_layer_1/gamma&Adam/v/custom_activation_layer_1/gammaAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/bias%Adam/m/custom_activation_layer_2/beta%Adam/v/custom_activation_layer_2/beta&Adam/m/custom_activation_layer_2/gamma&Adam/v/custom_activation_layer_2/gammaAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biastotalcount*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_136023��	
�U
�
A__inference_model_layer_call_and_return_conditional_losses_135361

inputs7
$dense_matmul_readvariableop_resource:	�4
%dense_biasadd_readvariableop_resource:	�B
3custom_activation_layer_mul_readvariableop_resource:	�B
3custom_activation_layer_sub_readvariableop_resource:	�:
&dense_1_matmul_readvariableop_resource:
��6
'dense_1_biasadd_readvariableop_resource:	�D
5custom_activation_layer_1_mul_readvariableop_resource:	�D
5custom_activation_layer_1_sub_readvariableop_resource:	�:
&dense_2_matmul_readvariableop_resource:
��6
'dense_2_biasadd_readvariableop_resource:	�D
5custom_activation_layer_2_mul_readvariableop_resource:	�D
5custom_activation_layer_2_sub_readvariableop_resource:	�9
&dense_3_matmul_readvariableop_resource:	�5
'dense_3_biasadd_readvariableop_resource:
identity��*custom_activation_layer/Add/ReadVariableOp�*custom_activation_layer/Mul/ReadVariableOp�*custom_activation_layer/Sub/ReadVariableOp�,custom_activation_layer_1/Add/ReadVariableOp�,custom_activation_layer_1/Mul/ReadVariableOp�,custom_activation_layer_1/Sub/ReadVariableOp�,custom_activation_layer_2/Add/ReadVariableOp�,custom_activation_layer_2/Mul/ReadVariableOp�,custom_activation_layer_2/Sub/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*custom_activation_layer/Mul/ReadVariableOpReadVariableOp3custom_activation_layer_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
custom_activation_layer/MulMul2custom_activation_layer/Mul/ReadVariableOp:value:0dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������~
custom_activation_layer/SigmoidSigmoidcustom_activation_layer/Mul:z:0*
T0*(
_output_shapes
:����������b
custom_activation_layer/Sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
*custom_activation_layer/Sub/ReadVariableOpReadVariableOp3custom_activation_layer_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
custom_activation_layer/SubSub&custom_activation_layer/Sub/x:output:02custom_activation_layer/Sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
custom_activation_layer/Mul_1Mul#custom_activation_layer/Sigmoid:y:0custom_activation_layer/Sub:z:0*
T0*(
_output_shapes
:�����������
*custom_activation_layer/Add/ReadVariableOpReadVariableOp3custom_activation_layer_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
custom_activation_layer/AddAddV22custom_activation_layer/Add/ReadVariableOp:value:0!custom_activation_layer/Mul_1:z:0*
T0*(
_output_shapes
:�����������
custom_activation_layer/Mul_2Mulcustom_activation_layer/Add:z:0dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_1/MatMulMatMul!custom_activation_layer/Mul_2:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,custom_activation_layer_1/Mul/ReadVariableOpReadVariableOp5custom_activation_layer_1_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
custom_activation_layer_1/MulMul4custom_activation_layer_1/Mul/ReadVariableOp:value:0dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
!custom_activation_layer_1/SigmoidSigmoid!custom_activation_layer_1/Mul:z:0*
T0*(
_output_shapes
:����������d
custom_activation_layer_1/Sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
,custom_activation_layer_1/Sub/ReadVariableOpReadVariableOp5custom_activation_layer_1_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
custom_activation_layer_1/SubSub(custom_activation_layer_1/Sub/x:output:04custom_activation_layer_1/Sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
custom_activation_layer_1/Mul_1Mul%custom_activation_layer_1/Sigmoid:y:0!custom_activation_layer_1/Sub:z:0*
T0*(
_output_shapes
:�����������
,custom_activation_layer_1/Add/ReadVariableOpReadVariableOp5custom_activation_layer_1_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
custom_activation_layer_1/AddAddV24custom_activation_layer_1/Add/ReadVariableOp:value:0#custom_activation_layer_1/Mul_1:z:0*
T0*(
_output_shapes
:�����������
custom_activation_layer_1/Mul_2Mul!custom_activation_layer_1/Add:z:0dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_2/MatMulMatMul#custom_activation_layer_1/Mul_2:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,custom_activation_layer_2/Mul/ReadVariableOpReadVariableOp5custom_activation_layer_2_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
custom_activation_layer_2/MulMul4custom_activation_layer_2/Mul/ReadVariableOp:value:0dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
!custom_activation_layer_2/SigmoidSigmoid!custom_activation_layer_2/Mul:z:0*
T0*(
_output_shapes
:����������d
custom_activation_layer_2/Sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
,custom_activation_layer_2/Sub/ReadVariableOpReadVariableOp5custom_activation_layer_2_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
custom_activation_layer_2/SubSub(custom_activation_layer_2/Sub/x:output:04custom_activation_layer_2/Sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
custom_activation_layer_2/Mul_1Mul%custom_activation_layer_2/Sigmoid:y:0!custom_activation_layer_2/Sub:z:0*
T0*(
_output_shapes
:�����������
,custom_activation_layer_2/Add/ReadVariableOpReadVariableOp5custom_activation_layer_2_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
custom_activation_layer_2/AddAddV24custom_activation_layer_2/Add/ReadVariableOp:value:0#custom_activation_layer_2/Mul_1:z:0*
T0*(
_output_shapes
:�����������
custom_activation_layer_2/Mul_2Mul!custom_activation_layer_2/Add:z:0dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_3/MatMulMatMul#custom_activation_layer_2/Mul_2:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^custom_activation_layer/Add/ReadVariableOp+^custom_activation_layer/Mul/ReadVariableOp+^custom_activation_layer/Sub/ReadVariableOp-^custom_activation_layer_1/Add/ReadVariableOp-^custom_activation_layer_1/Mul/ReadVariableOp-^custom_activation_layer_1/Sub/ReadVariableOp-^custom_activation_layer_2/Add/ReadVariableOp-^custom_activation_layer_2/Mul/ReadVariableOp-^custom_activation_layer_2/Sub/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2X
*custom_activation_layer/Add/ReadVariableOp*custom_activation_layer/Add/ReadVariableOp2X
*custom_activation_layer/Mul/ReadVariableOp*custom_activation_layer/Mul/ReadVariableOp2X
*custom_activation_layer/Sub/ReadVariableOp*custom_activation_layer/Sub/ReadVariableOp2\
,custom_activation_layer_1/Add/ReadVariableOp,custom_activation_layer_1/Add/ReadVariableOp2\
,custom_activation_layer_1/Mul/ReadVariableOp,custom_activation_layer_1/Mul/ReadVariableOp2\
,custom_activation_layer_1/Sub/ReadVariableOp,custom_activation_layer_1/Sub/ReadVariableOp2\
,custom_activation_layer_2/Add/ReadVariableOp,custom_activation_layer_2/Add/ReadVariableOp2\
,custom_activation_layer_2/Mul/ReadVariableOp,custom_activation_layer_2/Mul/ReadVariableOp2\
,custom_activation_layer_2/Sub/ReadVariableOp,custom_activation_layer_2/Sub/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
:__inference_custom_activation_layer_2_layer_call_fn_135541
x
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_custom_activation_layer_2_layer_call_and_return_conditional_losses_134851p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:����������

_user_specified_namex
�	
�
C__inference_dense_1_layer_call_and_return_conditional_losses_134791

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
:__inference_custom_activation_layer_1_layer_call_fn_135497
x
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_custom_activation_layer_1_layer_call_and_return_conditional_losses_134813p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:����������

_user_specified_namex
�)
�
A__inference_model_layer_call_and_return_conditional_losses_134874
input_1
dense_134754:	�
dense_134756:	�-
custom_activation_layer_134776:	�-
custom_activation_layer_134778:	�"
dense_1_134792:
��
dense_1_134794:	�/
 custom_activation_layer_1_134814:	�/
 custom_activation_layer_1_134816:	�"
dense_2_134830:
��
dense_2_134832:	�/
 custom_activation_layer_2_134852:	�/
 custom_activation_layer_2_134854:	�!
dense_3_134868:	�
dense_3_134870:
identity��/custom_activation_layer/StatefulPartitionedCall�1custom_activation_layer_1/StatefulPartitionedCall�1custom_activation_layer_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_134754dense_134756*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_134753�
/custom_activation_layer/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0custom_activation_layer_134776custom_activation_layer_134778*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_custom_activation_layer_layer_call_and_return_conditional_losses_134775�
dense_1/StatefulPartitionedCallStatefulPartitionedCall8custom_activation_layer/StatefulPartitionedCall:output:0dense_1_134792dense_1_134794*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_134791�
1custom_activation_layer_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 custom_activation_layer_1_134814 custom_activation_layer_1_134816*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_custom_activation_layer_1_layer_call_and_return_conditional_losses_134813�
dense_2/StatefulPartitionedCallStatefulPartitionedCall:custom_activation_layer_1/StatefulPartitionedCall:output:0dense_2_134830dense_2_134832*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_134829�
1custom_activation_layer_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0 custom_activation_layer_2_134852 custom_activation_layer_2_134854*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_custom_activation_layer_2_layer_call_and_return_conditional_losses_134851�
dense_3/StatefulPartitionedCallStatefulPartitionedCall:custom_activation_layer_2/StatefulPartitionedCall:output:0dense_3_134868dense_3_134870*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_134867w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp0^custom_activation_layer/StatefulPartitionedCall2^custom_activation_layer_1/StatefulPartitionedCall2^custom_activation_layer_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2b
/custom_activation_layer/StatefulPartitionedCall/custom_activation_layer/StatefulPartitionedCall2f
1custom_activation_layer_1/StatefulPartitionedCall1custom_activation_layer_1/StatefulPartitionedCall2f
1custom_activation_layer_2/StatefulPartitionedCall1custom_activation_layer_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
(__inference_dense_1_layer_call_fn_135478

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_134791p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_dense_2_layer_call_fn_135522

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_134829p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�U
�
A__inference_model_layer_call_and_return_conditional_losses_135425

inputs7
$dense_matmul_readvariableop_resource:	�4
%dense_biasadd_readvariableop_resource:	�B
3custom_activation_layer_mul_readvariableop_resource:	�B
3custom_activation_layer_sub_readvariableop_resource:	�:
&dense_1_matmul_readvariableop_resource:
��6
'dense_1_biasadd_readvariableop_resource:	�D
5custom_activation_layer_1_mul_readvariableop_resource:	�D
5custom_activation_layer_1_sub_readvariableop_resource:	�:
&dense_2_matmul_readvariableop_resource:
��6
'dense_2_biasadd_readvariableop_resource:	�D
5custom_activation_layer_2_mul_readvariableop_resource:	�D
5custom_activation_layer_2_sub_readvariableop_resource:	�9
&dense_3_matmul_readvariableop_resource:	�5
'dense_3_biasadd_readvariableop_resource:
identity��*custom_activation_layer/Add/ReadVariableOp�*custom_activation_layer/Mul/ReadVariableOp�*custom_activation_layer/Sub/ReadVariableOp�,custom_activation_layer_1/Add/ReadVariableOp�,custom_activation_layer_1/Mul/ReadVariableOp�,custom_activation_layer_1/Sub/ReadVariableOp�,custom_activation_layer_2/Add/ReadVariableOp�,custom_activation_layer_2/Mul/ReadVariableOp�,custom_activation_layer_2/Sub/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*custom_activation_layer/Mul/ReadVariableOpReadVariableOp3custom_activation_layer_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
custom_activation_layer/MulMul2custom_activation_layer/Mul/ReadVariableOp:value:0dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������~
custom_activation_layer/SigmoidSigmoidcustom_activation_layer/Mul:z:0*
T0*(
_output_shapes
:����������b
custom_activation_layer/Sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
*custom_activation_layer/Sub/ReadVariableOpReadVariableOp3custom_activation_layer_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
custom_activation_layer/SubSub&custom_activation_layer/Sub/x:output:02custom_activation_layer/Sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
custom_activation_layer/Mul_1Mul#custom_activation_layer/Sigmoid:y:0custom_activation_layer/Sub:z:0*
T0*(
_output_shapes
:�����������
*custom_activation_layer/Add/ReadVariableOpReadVariableOp3custom_activation_layer_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
custom_activation_layer/AddAddV22custom_activation_layer/Add/ReadVariableOp:value:0!custom_activation_layer/Mul_1:z:0*
T0*(
_output_shapes
:�����������
custom_activation_layer/Mul_2Mulcustom_activation_layer/Add:z:0dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_1/MatMulMatMul!custom_activation_layer/Mul_2:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,custom_activation_layer_1/Mul/ReadVariableOpReadVariableOp5custom_activation_layer_1_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
custom_activation_layer_1/MulMul4custom_activation_layer_1/Mul/ReadVariableOp:value:0dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
!custom_activation_layer_1/SigmoidSigmoid!custom_activation_layer_1/Mul:z:0*
T0*(
_output_shapes
:����������d
custom_activation_layer_1/Sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
,custom_activation_layer_1/Sub/ReadVariableOpReadVariableOp5custom_activation_layer_1_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
custom_activation_layer_1/SubSub(custom_activation_layer_1/Sub/x:output:04custom_activation_layer_1/Sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
custom_activation_layer_1/Mul_1Mul%custom_activation_layer_1/Sigmoid:y:0!custom_activation_layer_1/Sub:z:0*
T0*(
_output_shapes
:�����������
,custom_activation_layer_1/Add/ReadVariableOpReadVariableOp5custom_activation_layer_1_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
custom_activation_layer_1/AddAddV24custom_activation_layer_1/Add/ReadVariableOp:value:0#custom_activation_layer_1/Mul_1:z:0*
T0*(
_output_shapes
:�����������
custom_activation_layer_1/Mul_2Mul!custom_activation_layer_1/Add:z:0dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_2/MatMulMatMul#custom_activation_layer_1/Mul_2:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,custom_activation_layer_2/Mul/ReadVariableOpReadVariableOp5custom_activation_layer_2_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
custom_activation_layer_2/MulMul4custom_activation_layer_2/Mul/ReadVariableOp:value:0dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
!custom_activation_layer_2/SigmoidSigmoid!custom_activation_layer_2/Mul:z:0*
T0*(
_output_shapes
:����������d
custom_activation_layer_2/Sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
,custom_activation_layer_2/Sub/ReadVariableOpReadVariableOp5custom_activation_layer_2_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
custom_activation_layer_2/SubSub(custom_activation_layer_2/Sub/x:output:04custom_activation_layer_2/Sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
custom_activation_layer_2/Mul_1Mul%custom_activation_layer_2/Sigmoid:y:0!custom_activation_layer_2/Sub:z:0*
T0*(
_output_shapes
:�����������
,custom_activation_layer_2/Add/ReadVariableOpReadVariableOp5custom_activation_layer_2_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
custom_activation_layer_2/AddAddV24custom_activation_layer_2/Add/ReadVariableOp:value:0#custom_activation_layer_2/Mul_1:z:0*
T0*(
_output_shapes
:�����������
custom_activation_layer_2/Mul_2Mul!custom_activation_layer_2/Add:z:0dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_3/MatMulMatMul#custom_activation_layer_2/Mul_2:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^custom_activation_layer/Add/ReadVariableOp+^custom_activation_layer/Mul/ReadVariableOp+^custom_activation_layer/Sub/ReadVariableOp-^custom_activation_layer_1/Add/ReadVariableOp-^custom_activation_layer_1/Mul/ReadVariableOp-^custom_activation_layer_1/Sub/ReadVariableOp-^custom_activation_layer_2/Add/ReadVariableOp-^custom_activation_layer_2/Mul/ReadVariableOp-^custom_activation_layer_2/Sub/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2X
*custom_activation_layer/Add/ReadVariableOp*custom_activation_layer/Add/ReadVariableOp2X
*custom_activation_layer/Mul/ReadVariableOp*custom_activation_layer/Mul/ReadVariableOp2X
*custom_activation_layer/Sub/ReadVariableOp*custom_activation_layer/Sub/ReadVariableOp2\
,custom_activation_layer_1/Add/ReadVariableOp,custom_activation_layer_1/Add/ReadVariableOp2\
,custom_activation_layer_1/Mul/ReadVariableOp,custom_activation_layer_1/Mul/ReadVariableOp2\
,custom_activation_layer_1/Sub/ReadVariableOp,custom_activation_layer_1/Sub/ReadVariableOp2\
,custom_activation_layer_2/Add/ReadVariableOp,custom_activation_layer_2/Add/ReadVariableOp2\
,custom_activation_layer_2/Mul/ReadVariableOp,custom_activation_layer_2/Mul/ReadVariableOp2\
,custom_activation_layer_2/Sub/ReadVariableOp,custom_activation_layer_2/Sub/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_135231
input_1
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_134739o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
&__inference_model_layer_call_fn_134986
input_1
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_134955o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
8__inference_custom_activation_layer_layer_call_fn_135453
x
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_custom_activation_layer_layer_call_and_return_conditional_losses_134775p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:����������

_user_specified_namex
�(
�
A__inference_model_layer_call_and_return_conditional_losses_135027

inputs
dense_134991:	�
dense_134993:	�-
custom_activation_layer_134996:	�-
custom_activation_layer_134998:	�"
dense_1_135001:
��
dense_1_135003:	�/
 custom_activation_layer_1_135006:	�/
 custom_activation_layer_1_135008:	�"
dense_2_135011:
��
dense_2_135013:	�/
 custom_activation_layer_2_135016:	�/
 custom_activation_layer_2_135018:	�!
dense_3_135021:	�
dense_3_135023:
identity��/custom_activation_layer/StatefulPartitionedCall�1custom_activation_layer_1/StatefulPartitionedCall�1custom_activation_layer_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_134991dense_134993*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_134753�
/custom_activation_layer/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0custom_activation_layer_134996custom_activation_layer_134998*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_custom_activation_layer_layer_call_and_return_conditional_losses_134775�
dense_1/StatefulPartitionedCallStatefulPartitionedCall8custom_activation_layer/StatefulPartitionedCall:output:0dense_1_135001dense_1_135003*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_134791�
1custom_activation_layer_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 custom_activation_layer_1_135006 custom_activation_layer_1_135008*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_custom_activation_layer_1_layer_call_and_return_conditional_losses_134813�
dense_2/StatefulPartitionedCallStatefulPartitionedCall:custom_activation_layer_1/StatefulPartitionedCall:output:0dense_2_135011dense_2_135013*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_134829�
1custom_activation_layer_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0 custom_activation_layer_2_135016 custom_activation_layer_2_135018*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_custom_activation_layer_2_layer_call_and_return_conditional_losses_134851�
dense_3/StatefulPartitionedCallStatefulPartitionedCall:custom_activation_layer_2/StatefulPartitionedCall:output:0dense_3_135021dense_3_135023*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_134867w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp0^custom_activation_layer/StatefulPartitionedCall2^custom_activation_layer_1/StatefulPartitionedCall2^custom_activation_layer_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2b
/custom_activation_layer/StatefulPartitionedCall/custom_activation_layer/StatefulPartitionedCall2f
1custom_activation_layer_1/StatefulPartitionedCall1custom_activation_layer_1/StatefulPartitionedCall2f
1custom_activation_layer_2/StatefulPartitionedCall1custom_activation_layer_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
"__inference__traced_restore_136023
file_prefix0
assignvariableop_dense_kernel:	�,
assignvariableop_1_dense_bias:	�>
/assignvariableop_2_custom_activation_layer_beta:	�?
0assignvariableop_3_custom_activation_layer_gamma:	�5
!assignvariableop_4_dense_1_kernel:
��.
assignvariableop_5_dense_1_bias:	�@
1assignvariableop_6_custom_activation_layer_1_beta:	�A
2assignvariableop_7_custom_activation_layer_1_gamma:	�5
!assignvariableop_8_dense_2_kernel:
��.
assignvariableop_9_dense_2_bias:	�A
2assignvariableop_10_custom_activation_layer_2_beta:	�B
3assignvariableop_11_custom_activation_layer_2_gamma:	�5
"assignvariableop_12_dense_3_kernel:	�.
 assignvariableop_13_dense_3_bias:'
assignvariableop_14_iteration:	 +
!assignvariableop_15_learning_rate: :
'assignvariableop_16_adam_m_dense_kernel:	�:
'assignvariableop_17_adam_v_dense_kernel:	�4
%assignvariableop_18_adam_m_dense_bias:	�4
%assignvariableop_19_adam_v_dense_bias:	�F
7assignvariableop_20_adam_m_custom_activation_layer_beta:	�F
7assignvariableop_21_adam_v_custom_activation_layer_beta:	�G
8assignvariableop_22_adam_m_custom_activation_layer_gamma:	�G
8assignvariableop_23_adam_v_custom_activation_layer_gamma:	�=
)assignvariableop_24_adam_m_dense_1_kernel:
��=
)assignvariableop_25_adam_v_dense_1_kernel:
��6
'assignvariableop_26_adam_m_dense_1_bias:	�6
'assignvariableop_27_adam_v_dense_1_bias:	�H
9assignvariableop_28_adam_m_custom_activation_layer_1_beta:	�H
9assignvariableop_29_adam_v_custom_activation_layer_1_beta:	�I
:assignvariableop_30_adam_m_custom_activation_layer_1_gamma:	�I
:assignvariableop_31_adam_v_custom_activation_layer_1_gamma:	�=
)assignvariableop_32_adam_m_dense_2_kernel:
��=
)assignvariableop_33_adam_v_dense_2_kernel:
��6
'assignvariableop_34_adam_m_dense_2_bias:	�6
'assignvariableop_35_adam_v_dense_2_bias:	�H
9assignvariableop_36_adam_m_custom_activation_layer_2_beta:	�H
9assignvariableop_37_adam_v_custom_activation_layer_2_beta:	�I
:assignvariableop_38_adam_m_custom_activation_layer_2_gamma:	�I
:assignvariableop_39_adam_v_custom_activation_layer_2_gamma:	�<
)assignvariableop_40_adam_m_dense_3_kernel:	�<
)assignvariableop_41_adam_v_dense_3_kernel:	�5
'assignvariableop_42_adam_m_dense_3_bias:5
'assignvariableop_43_adam_v_dense_3_bias:#
assignvariableop_44_total: #
assignvariableop_45_count: 
identity_47��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*�
value�B�/B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::*=
dtypes3
12/	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp/assignvariableop_2_custom_activation_layer_betaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp0assignvariableop_3_custom_activation_layer_gammaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp1assignvariableop_6_custom_activation_layer_1_betaIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp2assignvariableop_7_custom_activation_layer_1_gammaIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_2_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_2_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp2assignvariableop_10_custom_activation_layer_2_betaIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp3assignvariableop_11_custom_activation_layer_2_gammaIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_3_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_3_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_iterationIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_learning_rateIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_m_dense_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_v_dense_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp%assignvariableop_18_adam_m_dense_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp%assignvariableop_19_adam_v_dense_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp7assignvariableop_20_adam_m_custom_activation_layer_betaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp7assignvariableop_21_adam_v_custom_activation_layer_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp8assignvariableop_22_adam_m_custom_activation_layer_gammaIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp8assignvariableop_23_adam_v_custom_activation_layer_gammaIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_m_dense_1_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_v_dense_1_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_m_dense_1_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_v_dense_1_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp9assignvariableop_28_adam_m_custom_activation_layer_1_betaIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp9assignvariableop_29_adam_v_custom_activation_layer_1_betaIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp:assignvariableop_30_adam_m_custom_activation_layer_1_gammaIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp:assignvariableop_31_adam_v_custom_activation_layer_1_gammaIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_m_dense_2_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_v_dense_2_kernelIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_m_dense_2_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_v_dense_2_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp9assignvariableop_36_adam_m_custom_activation_layer_2_betaIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp9assignvariableop_37_adam_v_custom_activation_layer_2_betaIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp:assignvariableop_38_adam_m_custom_activation_layer_2_gammaIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp:assignvariableop_39_adam_v_custom_activation_layer_2_gammaIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_m_dense_3_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_v_dense_3_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_m_dense_3_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp'assignvariableop_43_adam_v_dense_3_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOpassignvariableop_44_totalIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOpassignvariableop_45_countIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_46Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_47IdentityIdentity_46:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_47Identity_47:output:0*q
_input_shapes`
^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
C__inference_dense_1_layer_call_and_return_conditional_losses_135488

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�+
__inference__traced_save_135875
file_prefix6
#read_disablecopyonread_dense_kernel:	�2
#read_1_disablecopyonread_dense_bias:	�D
5read_2_disablecopyonread_custom_activation_layer_beta:	�E
6read_3_disablecopyonread_custom_activation_layer_gamma:	�;
'read_4_disablecopyonread_dense_1_kernel:
��4
%read_5_disablecopyonread_dense_1_bias:	�F
7read_6_disablecopyonread_custom_activation_layer_1_beta:	�G
8read_7_disablecopyonread_custom_activation_layer_1_gamma:	�;
'read_8_disablecopyonread_dense_2_kernel:
��4
%read_9_disablecopyonread_dense_2_bias:	�G
8read_10_disablecopyonread_custom_activation_layer_2_beta:	�H
9read_11_disablecopyonread_custom_activation_layer_2_gamma:	�;
(read_12_disablecopyonread_dense_3_kernel:	�4
&read_13_disablecopyonread_dense_3_bias:-
#read_14_disablecopyonread_iteration:	 1
'read_15_disablecopyonread_learning_rate: @
-read_16_disablecopyonread_adam_m_dense_kernel:	�@
-read_17_disablecopyonread_adam_v_dense_kernel:	�:
+read_18_disablecopyonread_adam_m_dense_bias:	�:
+read_19_disablecopyonread_adam_v_dense_bias:	�L
=read_20_disablecopyonread_adam_m_custom_activation_layer_beta:	�L
=read_21_disablecopyonread_adam_v_custom_activation_layer_beta:	�M
>read_22_disablecopyonread_adam_m_custom_activation_layer_gamma:	�M
>read_23_disablecopyonread_adam_v_custom_activation_layer_gamma:	�C
/read_24_disablecopyonread_adam_m_dense_1_kernel:
��C
/read_25_disablecopyonread_adam_v_dense_1_kernel:
��<
-read_26_disablecopyonread_adam_m_dense_1_bias:	�<
-read_27_disablecopyonread_adam_v_dense_1_bias:	�N
?read_28_disablecopyonread_adam_m_custom_activation_layer_1_beta:	�N
?read_29_disablecopyonread_adam_v_custom_activation_layer_1_beta:	�O
@read_30_disablecopyonread_adam_m_custom_activation_layer_1_gamma:	�O
@read_31_disablecopyonread_adam_v_custom_activation_layer_1_gamma:	�C
/read_32_disablecopyonread_adam_m_dense_2_kernel:
��C
/read_33_disablecopyonread_adam_v_dense_2_kernel:
��<
-read_34_disablecopyonread_adam_m_dense_2_bias:	�<
-read_35_disablecopyonread_adam_v_dense_2_bias:	�N
?read_36_disablecopyonread_adam_m_custom_activation_layer_2_beta:	�N
?read_37_disablecopyonread_adam_v_custom_activation_layer_2_beta:	�O
@read_38_disablecopyonread_adam_m_custom_activation_layer_2_gamma:	�O
@read_39_disablecopyonread_adam_v_custom_activation_layer_2_gamma:	�B
/read_40_disablecopyonread_adam_m_dense_3_kernel:	�B
/read_41_disablecopyonread_adam_v_dense_3_kernel:	�;
-read_42_disablecopyonread_adam_m_dense_3_bias:;
-read_43_disablecopyonread_adam_v_dense_3_bias:)
read_44_disablecopyonread_total: )
read_45_disablecopyonread_count: 
savev2_const
identity_93��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: u
Read/DisableCopyOnReadDisableCopyOnRead#read_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp#read_disablecopyonread_dense_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	�w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_dense_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_2/DisableCopyOnReadDisableCopyOnRead5read_2_disablecopyonread_custom_activation_layer_beta"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp5read_2_disablecopyonread_custom_activation_layer_beta^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_3/DisableCopyOnReadDisableCopyOnRead6read_3_disablecopyonread_custom_activation_layer_gamma"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp6read_3_disablecopyonread_custom_activation_layer_gamma^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�{
Read_4/DisableCopyOnReadDisableCopyOnRead'read_4_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp'read_4_disablecopyonread_dense_1_kernel^Read_4/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��y
Read_5/DisableCopyOnReadDisableCopyOnRead%read_5_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp%read_5_disablecopyonread_dense_1_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_6/DisableCopyOnReadDisableCopyOnRead7read_6_disablecopyonread_custom_activation_layer_1_beta"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp7read_6_disablecopyonread_custom_activation_layer_1_beta^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_7/DisableCopyOnReadDisableCopyOnRead8read_7_disablecopyonread_custom_activation_layer_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp8read_7_disablecopyonread_custom_activation_layer_1_gamma^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:�{
Read_8/DisableCopyOnReadDisableCopyOnRead'read_8_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp'read_8_disablecopyonread_dense_2_kernel^Read_8/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��y
Read_9/DisableCopyOnReadDisableCopyOnRead%read_9_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp%read_9_disablecopyonread_dense_2_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_10/DisableCopyOnReadDisableCopyOnRead8read_10_disablecopyonread_custom_activation_layer_2_beta"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp8read_10_disablecopyonread_custom_activation_layer_2_beta^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_11/DisableCopyOnReadDisableCopyOnRead9read_11_disablecopyonread_custom_activation_layer_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp9read_11_disablecopyonread_custom_activation_layer_2_gamma^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_12/DisableCopyOnReadDisableCopyOnRead(read_12_disablecopyonread_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp(read_12_disablecopyonread_dense_3_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:	�{
Read_13/DisableCopyOnReadDisableCopyOnRead&read_13_disablecopyonread_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp&read_13_disablecopyonread_dense_3_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_14/DisableCopyOnReadDisableCopyOnRead#read_14_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp#read_14_disablecopyonread_iteration^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_15/DisableCopyOnReadDisableCopyOnRead'read_15_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp'read_15_disablecopyonread_learning_rate^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_16/DisableCopyOnReadDisableCopyOnRead-read_16_disablecopyonread_adam_m_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp-read_16_disablecopyonread_adam_m_dense_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_17/DisableCopyOnReadDisableCopyOnRead-read_17_disablecopyonread_adam_v_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp-read_17_disablecopyonread_adam_v_dense_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_18/DisableCopyOnReadDisableCopyOnRead+read_18_disablecopyonread_adam_m_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp+read_18_disablecopyonread_adam_m_dense_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_19/DisableCopyOnReadDisableCopyOnRead+read_19_disablecopyonread_adam_v_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp+read_19_disablecopyonread_adam_v_dense_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_20/DisableCopyOnReadDisableCopyOnRead=read_20_disablecopyonread_adam_m_custom_activation_layer_beta"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp=read_20_disablecopyonread_adam_m_custom_activation_layer_beta^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_21/DisableCopyOnReadDisableCopyOnRead=read_21_disablecopyonread_adam_v_custom_activation_layer_beta"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp=read_21_disablecopyonread_adam_v_custom_activation_layer_beta^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_22/DisableCopyOnReadDisableCopyOnRead>read_22_disablecopyonread_adam_m_custom_activation_layer_gamma"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp>read_22_disablecopyonread_adam_m_custom_activation_layer_gamma^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_23/DisableCopyOnReadDisableCopyOnRead>read_23_disablecopyonread_adam_v_custom_activation_layer_gamma"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp>read_23_disablecopyonread_adam_v_custom_activation_layer_gamma^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_24/DisableCopyOnReadDisableCopyOnRead/read_24_disablecopyonread_adam_m_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp/read_24_disablecopyonread_adam_m_dense_1_kernel^Read_24/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_25/DisableCopyOnReadDisableCopyOnRead/read_25_disablecopyonread_adam_v_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp/read_25_disablecopyonread_adam_v_dense_1_kernel^Read_25/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_26/DisableCopyOnReadDisableCopyOnRead-read_26_disablecopyonread_adam_m_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp-read_26_disablecopyonread_adam_m_dense_1_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_27/DisableCopyOnReadDisableCopyOnRead-read_27_disablecopyonread_adam_v_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp-read_27_disablecopyonread_adam_v_dense_1_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_28/DisableCopyOnReadDisableCopyOnRead?read_28_disablecopyonread_adam_m_custom_activation_layer_1_beta"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp?read_28_disablecopyonread_adam_m_custom_activation_layer_1_beta^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_29/DisableCopyOnReadDisableCopyOnRead?read_29_disablecopyonread_adam_v_custom_activation_layer_1_beta"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp?read_29_disablecopyonread_adam_v_custom_activation_layer_1_beta^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_30/DisableCopyOnReadDisableCopyOnRead@read_30_disablecopyonread_adam_m_custom_activation_layer_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp@read_30_disablecopyonread_adam_m_custom_activation_layer_1_gamma^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_31/DisableCopyOnReadDisableCopyOnRead@read_31_disablecopyonread_adam_v_custom_activation_layer_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp@read_31_disablecopyonread_adam_v_custom_activation_layer_1_gamma^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_32/DisableCopyOnReadDisableCopyOnRead/read_32_disablecopyonread_adam_m_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp/read_32_disablecopyonread_adam_m_dense_2_kernel^Read_32/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_33/DisableCopyOnReadDisableCopyOnRead/read_33_disablecopyonread_adam_v_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp/read_33_disablecopyonread_adam_v_dense_2_kernel^Read_33/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_34/DisableCopyOnReadDisableCopyOnRead-read_34_disablecopyonread_adam_m_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp-read_34_disablecopyonread_adam_m_dense_2_bias^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_35/DisableCopyOnReadDisableCopyOnRead-read_35_disablecopyonread_adam_v_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp-read_35_disablecopyonread_adam_v_dense_2_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_36/DisableCopyOnReadDisableCopyOnRead?read_36_disablecopyonread_adam_m_custom_activation_layer_2_beta"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp?read_36_disablecopyonread_adam_m_custom_activation_layer_2_beta^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_37/DisableCopyOnReadDisableCopyOnRead?read_37_disablecopyonread_adam_v_custom_activation_layer_2_beta"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp?read_37_disablecopyonread_adam_v_custom_activation_layer_2_beta^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_38/DisableCopyOnReadDisableCopyOnRead@read_38_disablecopyonread_adam_m_custom_activation_layer_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp@read_38_disablecopyonread_adam_m_custom_activation_layer_2_gamma^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_39/DisableCopyOnReadDisableCopyOnRead@read_39_disablecopyonread_adam_v_custom_activation_layer_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp@read_39_disablecopyonread_adam_v_custom_activation_layer_2_gamma^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_40/DisableCopyOnReadDisableCopyOnRead/read_40_disablecopyonread_adam_m_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp/read_40_disablecopyonread_adam_m_dense_3_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_41/DisableCopyOnReadDisableCopyOnRead/read_41_disablecopyonread_adam_v_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp/read_41_disablecopyonread_adam_v_dense_3_kernel^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_42/DisableCopyOnReadDisableCopyOnRead-read_42_disablecopyonread_adam_m_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp-read_42_disablecopyonread_adam_m_dense_3_bias^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_43/DisableCopyOnReadDisableCopyOnRead-read_43_disablecopyonread_adam_v_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp-read_43_disablecopyonread_adam_v_dense_3_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_44/DisableCopyOnReadDisableCopyOnReadread_44_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOpread_44_disablecopyonread_total^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_45/DisableCopyOnReadDisableCopyOnReadread_45_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOpread_45_disablecopyonread_count^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*�
value�B�/B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *=
dtypes3
12/	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_92Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_93IdentityIdentity_92:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_93Identity_93:output:0*s
_input_shapesb
`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:/

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
C__inference_dense_2_layer_call_and_return_conditional_losses_134829

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_dense_layer_call_fn_135434

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_134753p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
C__inference_dense_2_layer_call_and_return_conditional_losses_135532

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_custom_activation_layer_layer_call_and_return_conditional_losses_135469
x*
mul_readvariableop_resource:	�*
sub_readvariableop_resource:	�
identity��Add/ReadVariableOp�Mul/ReadVariableOp�Sub/ReadVariableOpk
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0\
MulMulMul/ReadVariableOp:value:0x*
T0*(
_output_shapes
:����������N
SigmoidSigmoidMul:z:0*
T0*(
_output_shapes
:����������J
Sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
Sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0\
SubSubSub/x:output:0Sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:�U
Mul_1MulSigmoid:y:0Sub:z:0*
T0*(
_output_shapes
:����������k
Add/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0f
AddAddV2Add/ReadVariableOp:value:0	Mul_1:z:0*
T0*(
_output_shapes
:����������K
Mul_2MulAdd:z:0x*
T0*(
_output_shapes
:����������Y
IdentityIdentity	Mul_2:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^Add/ReadVariableOp^Mul/ReadVariableOp^Sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2(
Add/ReadVariableOpAdd/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2(
Sub/ReadVariableOpSub/ReadVariableOp:K G
(
_output_shapes
:����������

_user_specified_namex
�
�
U__inference_custom_activation_layer_1_layer_call_and_return_conditional_losses_134813
x*
mul_readvariableop_resource:	�*
sub_readvariableop_resource:	�
identity��Add/ReadVariableOp�Mul/ReadVariableOp�Sub/ReadVariableOpk
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0\
MulMulMul/ReadVariableOp:value:0x*
T0*(
_output_shapes
:����������N
SigmoidSigmoidMul:z:0*
T0*(
_output_shapes
:����������J
Sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
Sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0\
SubSubSub/x:output:0Sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:�U
Mul_1MulSigmoid:y:0Sub:z:0*
T0*(
_output_shapes
:����������k
Add/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0f
AddAddV2Add/ReadVariableOp:value:0	Mul_1:z:0*
T0*(
_output_shapes
:����������K
Mul_2MulAdd:z:0x*
T0*(
_output_shapes
:����������Y
IdentityIdentity	Mul_2:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^Add/ReadVariableOp^Mul/ReadVariableOp^Sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2(
Add/ReadVariableOpAdd/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2(
Sub/ReadVariableOpSub/ReadVariableOp:K G
(
_output_shapes
:����������

_user_specified_namex
�
�
&__inference_model_layer_call_fn_135264

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_134955o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
U__inference_custom_activation_layer_1_layer_call_and_return_conditional_losses_135513
x*
mul_readvariableop_resource:	�*
sub_readvariableop_resource:	�
identity��Add/ReadVariableOp�Mul/ReadVariableOp�Sub/ReadVariableOpk
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0\
MulMulMul/ReadVariableOp:value:0x*
T0*(
_output_shapes
:����������N
SigmoidSigmoidMul:z:0*
T0*(
_output_shapes
:����������J
Sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
Sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0\
SubSubSub/x:output:0Sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:�U
Mul_1MulSigmoid:y:0Sub:z:0*
T0*(
_output_shapes
:����������k
Add/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0f
AddAddV2Add/ReadVariableOp:value:0	Mul_1:z:0*
T0*(
_output_shapes
:����������K
Mul_2MulAdd:z:0x*
T0*(
_output_shapes
:����������Y
IdentityIdentity	Mul_2:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^Add/ReadVariableOp^Mul/ReadVariableOp^Sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2(
Add/ReadVariableOpAdd/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2(
Sub/ReadVariableOpSub/ReadVariableOp:K G
(
_output_shapes
:����������

_user_specified_namex
�	
�
C__inference_dense_3_layer_call_and_return_conditional_losses_134867

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_custom_activation_layer_layer_call_and_return_conditional_losses_134775
x*
mul_readvariableop_resource:	�*
sub_readvariableop_resource:	�
identity��Add/ReadVariableOp�Mul/ReadVariableOp�Sub/ReadVariableOpk
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0\
MulMulMul/ReadVariableOp:value:0x*
T0*(
_output_shapes
:����������N
SigmoidSigmoidMul:z:0*
T0*(
_output_shapes
:����������J
Sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
Sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0\
SubSubSub/x:output:0Sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:�U
Mul_1MulSigmoid:y:0Sub:z:0*
T0*(
_output_shapes
:����������k
Add/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0f
AddAddV2Add/ReadVariableOp:value:0	Mul_1:z:0*
T0*(
_output_shapes
:����������K
Mul_2MulAdd:z:0x*
T0*(
_output_shapes
:����������Y
IdentityIdentity	Mul_2:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^Add/ReadVariableOp^Mul/ReadVariableOp^Sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2(
Add/ReadVariableOpAdd/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2(
Sub/ReadVariableOpSub/ReadVariableOp:K G
(
_output_shapes
:����������

_user_specified_namex
�
�
U__inference_custom_activation_layer_2_layer_call_and_return_conditional_losses_135557
x*
mul_readvariableop_resource:	�*
sub_readvariableop_resource:	�
identity��Add/ReadVariableOp�Mul/ReadVariableOp�Sub/ReadVariableOpk
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0\
MulMulMul/ReadVariableOp:value:0x*
T0*(
_output_shapes
:����������N
SigmoidSigmoidMul:z:0*
T0*(
_output_shapes
:����������J
Sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
Sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0\
SubSubSub/x:output:0Sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:�U
Mul_1MulSigmoid:y:0Sub:z:0*
T0*(
_output_shapes
:����������k
Add/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0f
AddAddV2Add/ReadVariableOp:value:0	Mul_1:z:0*
T0*(
_output_shapes
:����������K
Mul_2MulAdd:z:0x*
T0*(
_output_shapes
:����������Y
IdentityIdentity	Mul_2:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^Add/ReadVariableOp^Mul/ReadVariableOp^Sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2(
Add/ReadVariableOpAdd/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2(
Sub/ReadVariableOpSub/ReadVariableOp:K G
(
_output_shapes
:����������

_user_specified_namex
�(
�
A__inference_model_layer_call_and_return_conditional_losses_134955

inputs
dense_134919:	�
dense_134921:	�-
custom_activation_layer_134924:	�-
custom_activation_layer_134926:	�"
dense_1_134929:
��
dense_1_134931:	�/
 custom_activation_layer_1_134934:	�/
 custom_activation_layer_1_134936:	�"
dense_2_134939:
��
dense_2_134941:	�/
 custom_activation_layer_2_134944:	�/
 custom_activation_layer_2_134946:	�!
dense_3_134949:	�
dense_3_134951:
identity��/custom_activation_layer/StatefulPartitionedCall�1custom_activation_layer_1/StatefulPartitionedCall�1custom_activation_layer_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_134919dense_134921*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_134753�
/custom_activation_layer/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0custom_activation_layer_134924custom_activation_layer_134926*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_custom_activation_layer_layer_call_and_return_conditional_losses_134775�
dense_1/StatefulPartitionedCallStatefulPartitionedCall8custom_activation_layer/StatefulPartitionedCall:output:0dense_1_134929dense_1_134931*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_134791�
1custom_activation_layer_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 custom_activation_layer_1_134934 custom_activation_layer_1_134936*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_custom_activation_layer_1_layer_call_and_return_conditional_losses_134813�
dense_2/StatefulPartitionedCallStatefulPartitionedCall:custom_activation_layer_1/StatefulPartitionedCall:output:0dense_2_134939dense_2_134941*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_134829�
1custom_activation_layer_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0 custom_activation_layer_2_134944 custom_activation_layer_2_134946*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_custom_activation_layer_2_layer_call_and_return_conditional_losses_134851�
dense_3/StatefulPartitionedCallStatefulPartitionedCall:custom_activation_layer_2/StatefulPartitionedCall:output:0dense_3_134949dense_3_134951*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_134867w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp0^custom_activation_layer/StatefulPartitionedCall2^custom_activation_layer_1/StatefulPartitionedCall2^custom_activation_layer_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2b
/custom_activation_layer/StatefulPartitionedCall/custom_activation_layer/StatefulPartitionedCall2f
1custom_activation_layer_1/StatefulPartitionedCall1custom_activation_layer_1/StatefulPartitionedCall2f
1custom_activation_layer_2/StatefulPartitionedCall1custom_activation_layer_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_135058
input_1
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_135027o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�^
�
!__inference__wrapped_model_134739
input_1=
*model_dense_matmul_readvariableop_resource:	�:
+model_dense_biasadd_readvariableop_resource:	�H
9model_custom_activation_layer_mul_readvariableop_resource:	�H
9model_custom_activation_layer_sub_readvariableop_resource:	�@
,model_dense_1_matmul_readvariableop_resource:
��<
-model_dense_1_biasadd_readvariableop_resource:	�J
;model_custom_activation_layer_1_mul_readvariableop_resource:	�J
;model_custom_activation_layer_1_sub_readvariableop_resource:	�@
,model_dense_2_matmul_readvariableop_resource:
��<
-model_dense_2_biasadd_readvariableop_resource:	�J
;model_custom_activation_layer_2_mul_readvariableop_resource:	�J
;model_custom_activation_layer_2_sub_readvariableop_resource:	�?
,model_dense_3_matmul_readvariableop_resource:	�;
-model_dense_3_biasadd_readvariableop_resource:
identity��0model/custom_activation_layer/Add/ReadVariableOp�0model/custom_activation_layer/Mul/ReadVariableOp�0model/custom_activation_layer/Sub/ReadVariableOp�2model/custom_activation_layer_1/Add/ReadVariableOp�2model/custom_activation_layer_1/Mul/ReadVariableOp�2model/custom_activation_layer_1/Sub/ReadVariableOp�2model/custom_activation_layer_2/Add/ReadVariableOp�2model/custom_activation_layer_2/Mul/ReadVariableOp�2model/custom_activation_layer_2/Sub/ReadVariableOp�"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�$model/dense_2/BiasAdd/ReadVariableOp�#model/dense_2/MatMul/ReadVariableOp�$model/dense_3/BiasAdd/ReadVariableOp�#model/dense_3/MatMul/ReadVariableOp�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/dense/MatMulMatMulinput_1)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
0model/custom_activation_layer/Mul/ReadVariableOpReadVariableOp9model_custom_activation_layer_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!model/custom_activation_layer/MulMul8model/custom_activation_layer/Mul/ReadVariableOp:value:0model/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
%model/custom_activation_layer/SigmoidSigmoid%model/custom_activation_layer/Mul:z:0*
T0*(
_output_shapes
:����������h
#model/custom_activation_layer/Sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
0model/custom_activation_layer/Sub/ReadVariableOpReadVariableOp9model_custom_activation_layer_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!model/custom_activation_layer/SubSub,model/custom_activation_layer/Sub/x:output:08model/custom_activation_layer/Sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
#model/custom_activation_layer/Mul_1Mul)model/custom_activation_layer/Sigmoid:y:0%model/custom_activation_layer/Sub:z:0*
T0*(
_output_shapes
:�����������
0model/custom_activation_layer/Add/ReadVariableOpReadVariableOp9model_custom_activation_layer_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!model/custom_activation_layer/AddAddV28model/custom_activation_layer/Add/ReadVariableOp:value:0'model/custom_activation_layer/Mul_1:z:0*
T0*(
_output_shapes
:�����������
#model/custom_activation_layer/Mul_2Mul%model/custom_activation_layer/Add:z:0model/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/dense_1/MatMulMatMul'model/custom_activation_layer/Mul_2:z:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2model/custom_activation_layer_1/Mul/ReadVariableOpReadVariableOp;model_custom_activation_layer_1_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#model/custom_activation_layer_1/MulMul:model/custom_activation_layer_1/Mul/ReadVariableOp:value:0model/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
'model/custom_activation_layer_1/SigmoidSigmoid'model/custom_activation_layer_1/Mul:z:0*
T0*(
_output_shapes
:����������j
%model/custom_activation_layer_1/Sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
2model/custom_activation_layer_1/Sub/ReadVariableOpReadVariableOp;model_custom_activation_layer_1_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#model/custom_activation_layer_1/SubSub.model/custom_activation_layer_1/Sub/x:output:0:model/custom_activation_layer_1/Sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
%model/custom_activation_layer_1/Mul_1Mul+model/custom_activation_layer_1/Sigmoid:y:0'model/custom_activation_layer_1/Sub:z:0*
T0*(
_output_shapes
:�����������
2model/custom_activation_layer_1/Add/ReadVariableOpReadVariableOp;model_custom_activation_layer_1_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#model/custom_activation_layer_1/AddAddV2:model/custom_activation_layer_1/Add/ReadVariableOp:value:0)model/custom_activation_layer_1/Mul_1:z:0*
T0*(
_output_shapes
:�����������
%model/custom_activation_layer_1/Mul_2Mul'model/custom_activation_layer_1/Add:z:0model/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/dense_2/MatMulMatMul)model/custom_activation_layer_1/Mul_2:z:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2model/custom_activation_layer_2/Mul/ReadVariableOpReadVariableOp;model_custom_activation_layer_2_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#model/custom_activation_layer_2/MulMul:model/custom_activation_layer_2/Mul/ReadVariableOp:value:0model/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
'model/custom_activation_layer_2/SigmoidSigmoid'model/custom_activation_layer_2/Mul:z:0*
T0*(
_output_shapes
:����������j
%model/custom_activation_layer_2/Sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
2model/custom_activation_layer_2/Sub/ReadVariableOpReadVariableOp;model_custom_activation_layer_2_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#model/custom_activation_layer_2/SubSub.model/custom_activation_layer_2/Sub/x:output:0:model/custom_activation_layer_2/Sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
%model/custom_activation_layer_2/Mul_1Mul+model/custom_activation_layer_2/Sigmoid:y:0'model/custom_activation_layer_2/Sub:z:0*
T0*(
_output_shapes
:�����������
2model/custom_activation_layer_2/Add/ReadVariableOpReadVariableOp;model_custom_activation_layer_2_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#model/custom_activation_layer_2/AddAddV2:model/custom_activation_layer_2/Add/ReadVariableOp:value:0)model/custom_activation_layer_2/Mul_1:z:0*
T0*(
_output_shapes
:�����������
%model/custom_activation_layer_2/Mul_2Mul'model/custom_activation_layer_2/Add:z:0model/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/dense_3/MatMulMatMul)model/custom_activation_layer_2/Mul_2:z:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������m
IdentityIdentitymodel/dense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp1^model/custom_activation_layer/Add/ReadVariableOp1^model/custom_activation_layer/Mul/ReadVariableOp1^model/custom_activation_layer/Sub/ReadVariableOp3^model/custom_activation_layer_1/Add/ReadVariableOp3^model/custom_activation_layer_1/Mul/ReadVariableOp3^model/custom_activation_layer_1/Sub/ReadVariableOp3^model/custom_activation_layer_2/Add/ReadVariableOp3^model/custom_activation_layer_2/Mul/ReadVariableOp3^model/custom_activation_layer_2/Sub/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2d
0model/custom_activation_layer/Add/ReadVariableOp0model/custom_activation_layer/Add/ReadVariableOp2d
0model/custom_activation_layer/Mul/ReadVariableOp0model/custom_activation_layer/Mul/ReadVariableOp2d
0model/custom_activation_layer/Sub/ReadVariableOp0model/custom_activation_layer/Sub/ReadVariableOp2h
2model/custom_activation_layer_1/Add/ReadVariableOp2model/custom_activation_layer_1/Add/ReadVariableOp2h
2model/custom_activation_layer_1/Mul/ReadVariableOp2model/custom_activation_layer_1/Mul/ReadVariableOp2h
2model/custom_activation_layer_1/Sub/ReadVariableOp2model/custom_activation_layer_1/Sub/ReadVariableOp2h
2model/custom_activation_layer_2/Add/ReadVariableOp2model/custom_activation_layer_2/Add/ReadVariableOp2h
2model/custom_activation_layer_2/Mul/ReadVariableOp2model/custom_activation_layer_2/Mul/ReadVariableOp2h
2model/custom_activation_layer_2/Sub/ReadVariableOp2model/custom_activation_layer_2/Sub/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
U__inference_custom_activation_layer_2_layer_call_and_return_conditional_losses_134851
x*
mul_readvariableop_resource:	�*
sub_readvariableop_resource:	�
identity��Add/ReadVariableOp�Mul/ReadVariableOp�Sub/ReadVariableOpk
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0\
MulMulMul/ReadVariableOp:value:0x*
T0*(
_output_shapes
:����������N
SigmoidSigmoidMul:z:0*
T0*(
_output_shapes
:����������J
Sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
Sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0\
SubSubSub/x:output:0Sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:�U
Mul_1MulSigmoid:y:0Sub:z:0*
T0*(
_output_shapes
:����������k
Add/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0f
AddAddV2Add/ReadVariableOp:value:0	Mul_1:z:0*
T0*(
_output_shapes
:����������K
Mul_2MulAdd:z:0x*
T0*(
_output_shapes
:����������Y
IdentityIdentity	Mul_2:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^Add/ReadVariableOp^Mul/ReadVariableOp^Sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2(
Add/ReadVariableOpAdd/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2(
Sub/ReadVariableOpSub/ReadVariableOp:K G
(
_output_shapes
:����������

_user_specified_namex
�	
�
A__inference_dense_layer_call_and_return_conditional_losses_135444

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
A__inference_dense_layer_call_and_return_conditional_losses_134753

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_135297

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_135027o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
C__inference_dense_3_layer_call_and_return_conditional_losses_135576

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_dense_3_layer_call_fn_135566

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_134867o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�)
�
A__inference_model_layer_call_and_return_conditional_losses_134913
input_1
dense_134877:	�
dense_134879:	�-
custom_activation_layer_134882:	�-
custom_activation_layer_134884:	�"
dense_1_134887:
��
dense_1_134889:	�/
 custom_activation_layer_1_134892:	�/
 custom_activation_layer_1_134894:	�"
dense_2_134897:
��
dense_2_134899:	�/
 custom_activation_layer_2_134902:	�/
 custom_activation_layer_2_134904:	�!
dense_3_134907:	�
dense_3_134909:
identity��/custom_activation_layer/StatefulPartitionedCall�1custom_activation_layer_1/StatefulPartitionedCall�1custom_activation_layer_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_134877dense_134879*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_134753�
/custom_activation_layer/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0custom_activation_layer_134882custom_activation_layer_134884*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_custom_activation_layer_layer_call_and_return_conditional_losses_134775�
dense_1/StatefulPartitionedCallStatefulPartitionedCall8custom_activation_layer/StatefulPartitionedCall:output:0dense_1_134887dense_1_134889*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_134791�
1custom_activation_layer_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 custom_activation_layer_1_134892 custom_activation_layer_1_134894*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_custom_activation_layer_1_layer_call_and_return_conditional_losses_134813�
dense_2/StatefulPartitionedCallStatefulPartitionedCall:custom_activation_layer_1/StatefulPartitionedCall:output:0dense_2_134897dense_2_134899*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_134829�
1custom_activation_layer_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0 custom_activation_layer_2_134902 custom_activation_layer_2_134904*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_custom_activation_layer_2_layer_call_and_return_conditional_losses_134851�
dense_3/StatefulPartitionedCallStatefulPartitionedCall:custom_activation_layer_2/StatefulPartitionedCall:output:0dense_3_134907dense_3_134909*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_134867w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp0^custom_activation_layer/StatefulPartitionedCall2^custom_activation_layer_1/StatefulPartitionedCall2^custom_activation_layer_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2b
/custom_activation_layer/StatefulPartitionedCall/custom_activation_layer/StatefulPartitionedCall2f
1custom_activation_layer_1/StatefulPartitionedCall1custom_activation_layer_1/StatefulPartitionedCall2f
1custom_activation_layer_2/StatefulPartitionedCall1custom_activation_layer_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������;
dense_30
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 beta
	!gamma"
_tf_keras_layer
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias"
_tf_keras_layer
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0beta
	1gamma"
_tf_keras_layer
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias"
_tf_keras_layer
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@beta
	Agamma"
_tf_keras_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

Hkernel
Ibias"
_tf_keras_layer
�
0
1
 2
!3
(4
)5
06
17
88
99
@10
A11
H12
I13"
trackable_list_wrapper
�
0
1
 2
!3
(4
)5
06
17
88
99
@10
A11
H12
I13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_32�
&__inference_model_layer_call_fn_134986
&__inference_model_layer_call_fn_135058
&__inference_model_layer_call_fn_135264
&__inference_model_layer_call_fn_135297�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zOtrace_0zPtrace_1zQtrace_2zRtrace_3
�
Strace_0
Ttrace_1
Utrace_2
Vtrace_32�
A__inference_model_layer_call_and_return_conditional_losses_134874
A__inference_model_layer_call_and_return_conditional_losses_134913
A__inference_model_layer_call_and_return_conditional_losses_135361
A__inference_model_layer_call_and_return_conditional_losses_135425�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zStrace_0zTtrace_1zUtrace_2zVtrace_3
�B�
!__inference__wrapped_model_134739input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
W
_variables
X_iterations
Y_learning_rate
Z_index_dict
[
_momentums
\_velocities
]_update_step_xla"
experimentalOptimizer
,
^serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
dtrace_02�
&__inference_dense_layer_call_fn_135434�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zdtrace_0
�
etrace_02�
A__inference_dense_layer_call_and_return_conditional_losses_135444�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zetrace_0
:	�2dense/kernel
:�2
dense/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
ktrace_02�
8__inference_custom_activation_layer_layer_call_fn_135453�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zktrace_0
�
ltrace_02�
S__inference_custom_activation_layer_layer_call_and_return_conditional_losses_135469�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zltrace_0
+:)�2custom_activation_layer/beta
,:*�2custom_activation_layer/gamma
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�
rtrace_02�
(__inference_dense_1_layer_call_fn_135478�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zrtrace_0
�
strace_02�
C__inference_dense_1_layer_call_and_return_conditional_losses_135488�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zstrace_0
": 
��2dense_1/kernel
:�2dense_1/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�
ytrace_02�
:__inference_custom_activation_layer_1_layer_call_fn_135497�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zytrace_0
�
ztrace_02�
U__inference_custom_activation_layer_1_layer_call_and_return_conditional_losses_135513�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zztrace_0
-:+�2custom_activation_layer_1/beta
.:,�2custom_activation_layer_1/gamma
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
�
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_2_layer_call_fn_135522�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_2_layer_call_and_return_conditional_losses_135532�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 
��2dense_2/kernel
:�2dense_2/bias
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
:__inference_custom_activation_layer_2_layer_call_fn_135541�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
U__inference_custom_activation_layer_2_layer_call_and_return_conditional_losses_135557�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+�2custom_activation_layer_2/beta
.:,�2custom_activation_layer_2/gamma
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_3_layer_call_fn_135566�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_3_layer_call_and_return_conditional_losses_135576�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:	�2dense_3/kernel
:2dense_3/bias
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_model_layer_call_fn_134986input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_model_layer_call_fn_135058input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_model_layer_call_fn_135264inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_model_layer_call_fn_135297inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_model_layer_call_and_return_conditional_losses_134874input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_model_layer_call_and_return_conditional_losses_134913input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_model_layer_call_and_return_conditional_losses_135361inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_model_layer_call_and_return_conditional_losses_135425inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
X0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_135231input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
&__inference_dense_layer_call_fn_135434inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_layer_call_and_return_conditional_losses_135444inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
8__inference_custom_activation_layer_layer_call_fn_135453x"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_custom_activation_layer_layer_call_and_return_conditional_losses_135469x"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
(__inference_dense_1_layer_call_fn_135478inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_1_layer_call_and_return_conditional_losses_135488inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
:__inference_custom_activation_layer_1_layer_call_fn_135497x"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_custom_activation_layer_1_layer_call_and_return_conditional_losses_135513x"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
(__inference_dense_2_layer_call_fn_135522inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_2_layer_call_and_return_conditional_losses_135532inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
:__inference_custom_activation_layer_2_layer_call_fn_135541x"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_custom_activation_layer_2_layer_call_and_return_conditional_losses_135557x"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
(__inference_dense_3_layer_call_fn_135566inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_3_layer_call_and_return_conditional_losses_135576inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
$:"	�2Adam/m/dense/kernel
$:"	�2Adam/v/dense/kernel
:�2Adam/m/dense/bias
:�2Adam/v/dense/bias
0:.�2#Adam/m/custom_activation_layer/beta
0:.�2#Adam/v/custom_activation_layer/beta
1:/�2$Adam/m/custom_activation_layer/gamma
1:/�2$Adam/v/custom_activation_layer/gamma
':%
��2Adam/m/dense_1/kernel
':%
��2Adam/v/dense_1/kernel
 :�2Adam/m/dense_1/bias
 :�2Adam/v/dense_1/bias
2:0�2%Adam/m/custom_activation_layer_1/beta
2:0�2%Adam/v/custom_activation_layer_1/beta
3:1�2&Adam/m/custom_activation_layer_1/gamma
3:1�2&Adam/v/custom_activation_layer_1/gamma
':%
��2Adam/m/dense_2/kernel
':%
��2Adam/v/dense_2/kernel
 :�2Adam/m/dense_2/bias
 :�2Adam/v/dense_2/bias
2:0�2%Adam/m/custom_activation_layer_2/beta
2:0�2%Adam/v/custom_activation_layer_2/beta
3:1�2&Adam/m/custom_activation_layer_2/gamma
3:1�2&Adam/v/custom_activation_layer_2/gamma
&:$	�2Adam/m/dense_3/kernel
&:$	�2Adam/v/dense_3/kernel
:2Adam/m/dense_3/bias
:2Adam/v/dense_3/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count�
!__inference__wrapped_model_134739u !()0189@AHI0�-
&�#
!�
input_1���������
� "1�.
,
dense_3!�
dense_3����������
U__inference_custom_activation_layer_1_layer_call_and_return_conditional_losses_135513`01+�(
!�
�
x����������
� "-�*
#� 
tensor_0����������
� �
:__inference_custom_activation_layer_1_layer_call_fn_135497U01+�(
!�
�
x����������
� ""�
unknown�����������
U__inference_custom_activation_layer_2_layer_call_and_return_conditional_losses_135557`@A+�(
!�
�
x����������
� "-�*
#� 
tensor_0����������
� �
:__inference_custom_activation_layer_2_layer_call_fn_135541U@A+�(
!�
�
x����������
� ""�
unknown�����������
S__inference_custom_activation_layer_layer_call_and_return_conditional_losses_135469` !+�(
!�
�
x����������
� "-�*
#� 
tensor_0����������
� �
8__inference_custom_activation_layer_layer_call_fn_135453U !+�(
!�
�
x����������
� ""�
unknown�����������
C__inference_dense_1_layer_call_and_return_conditional_losses_135488e()0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_dense_1_layer_call_fn_135478Z()0�-
&�#
!�
inputs����������
� ""�
unknown�����������
C__inference_dense_2_layer_call_and_return_conditional_losses_135532e890�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_dense_2_layer_call_fn_135522Z890�-
&�#
!�
inputs����������
� ""�
unknown�����������
C__inference_dense_3_layer_call_and_return_conditional_losses_135576dHI0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
(__inference_dense_3_layer_call_fn_135566YHI0�-
&�#
!�
inputs����������
� "!�
unknown����������
A__inference_dense_layer_call_and_return_conditional_losses_135444d/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
&__inference_dense_layer_call_fn_135434Y/�,
%�"
 �
inputs���������
� ""�
unknown�����������
A__inference_model_layer_call_and_return_conditional_losses_134874x !()0189@AHI8�5
.�+
!�
input_1���������
p

 
� ",�)
"�
tensor_0���������
� �
A__inference_model_layer_call_and_return_conditional_losses_134913x !()0189@AHI8�5
.�+
!�
input_1���������
p 

 
� ",�)
"�
tensor_0���������
� �
A__inference_model_layer_call_and_return_conditional_losses_135361w !()0189@AHI7�4
-�*
 �
inputs���������
p

 
� ",�)
"�
tensor_0���������
� �
A__inference_model_layer_call_and_return_conditional_losses_135425w !()0189@AHI7�4
-�*
 �
inputs���������
p 

 
� ",�)
"�
tensor_0���������
� �
&__inference_model_layer_call_fn_134986m !()0189@AHI8�5
.�+
!�
input_1���������
p

 
� "!�
unknown����������
&__inference_model_layer_call_fn_135058m !()0189@AHI8�5
.�+
!�
input_1���������
p 

 
� "!�
unknown����������
&__inference_model_layer_call_fn_135264l !()0189@AHI7�4
-�*
 �
inputs���������
p

 
� "!�
unknown����������
&__inference_model_layer_call_fn_135297l !()0189@AHI7�4
-�*
 �
inputs���������
p 

 
� "!�
unknown����������
$__inference_signature_wrapper_135231� !()0189@AHI;�8
� 
1�.
,
input_1!�
input_1���������"1�.
,
dense_3!�
dense_3���������