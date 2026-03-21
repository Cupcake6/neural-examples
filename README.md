# 1. Dot classification
![An example of using the library to classify dots](images/dot-classification-1.png)

Press LMB (the left mouse button) to place orange dots and RMB (the right mouse button) to place blue dots. The network will learn their placement pattern in real time. You may tweak the network in the `dot-classification/src/main.rs` file.

Run this command from the project root to launch this example:

``` shell
cargo run -p dot-classification --release
```

Release mode is preferred for speed.

# 2. Function approximation
![Network tries to approximate a spiky function](images/function-approximation-1.png)

Network tries to approximate `x % 3.0` (which yields negative values for negative inputs). Red = true function, blue = network output.
You can try out different functions by tweaking `function-approximation/src/main.rs`.

To launch this example, run:
``` shell
cargo run -p function-approximation --release
```