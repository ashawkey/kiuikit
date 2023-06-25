import kiui

kiui.try_import("os", "os", True)
print(os)

kiui.env("data", verbose=False)
print(globals())

try:
    kiui.env("NotAPack", verbose=True)
except Exception as e:
    print(e)

x = np.random.rand(4, 4, 3)
# kiui.vis.plot_image(x)

y = kiui.op.normalize(x)
print(y)
