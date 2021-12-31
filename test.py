import kiui

kiui.try_import('os', 'os', True)
print(os)

kiui.env(verbose=True)
print(globals())

kiui.env('torch', verbose=True)
print(globals())

kiui.env('notapack', verbose=True)