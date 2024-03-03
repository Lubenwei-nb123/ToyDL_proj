is_simple = False

if is_simple:
    from toydl.core_simple import Variable
    from toydl.core_simple import Function
    from toydl.core_simple import using_config
    from toydl.core_simple import no_grad
    from toydl.core_simple import as_array
    from toydl.core_simple import as_variable
else:
    from toydl.core import Variable
    from toydl.core import Function
    from toydl.core import using_config
    from toydl.core import no_grad
    from toydl.core import as_array
    from toydl.core import as_variable
    from toydl.core import Parameter

