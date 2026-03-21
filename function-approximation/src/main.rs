use macroquad::prelude::*;
use neural::prelude::*;
use nalgebra::dvector;

const SCREEN_WIDTH: i32 = 1200;
const SCREEN_HEIGHT: i32 = 800;
const ZOOM: f32 = 80.0;
const LINE_THICKNESS: f32 = 3.0;
const LINE_COLOR: Color = RED;
const APPROXIMATION_LINE_COLOR: Color = BLUE;
const BACKGROUND_COLOR: Color = Color::from_rgba(17, 17, 27, 255);
const GRID_THICKNESS: f32 = 2.0;
const GRID_COLOR: Color = Color::from_rgba(30, 30, 46, 255);

// Edit this to see other functions
fn function_to_approximate(x: f32) -> f32 {
    (2.0 * x).sin()
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Function Approximation".to_string(),
        window_width: SCREEN_WIDTH,
        window_height: SCREEN_HEIGHT,
        window_resizable: false,
        ..Default::default()
    }
}

fn draw_grid() {
    let mut gx: f32 = SCREEN_WIDTH as f32 / 2.0 + ((SCREEN_WIDTH as f32 / 2.0) / ZOOM).floor() * ZOOM;
    while gx >= 0.0 {
        draw_line(gx, 0.0, gx, SCREEN_HEIGHT as f32, GRID_THICKNESS, GRID_COLOR);
        gx -= ZOOM;
    }

    let mut gy = SCREEN_HEIGHT as f32 / 2.0 + ((SCREEN_HEIGHT as f32 / 2.0) / ZOOM).floor() * ZOOM;
    while gy >= 0.0 {
        draw_line(0.0, gy, SCREEN_WIDTH as f32, gy, GRID_THICKNESS, GRID_COLOR);
        gy -= ZOOM;
    }
}

fn draw_function(f: &impl Fn(f32) -> f32, color: Color) {
    let mut last_ys: Option<f32> = None;

    for xs in 0..(SCREEN_WIDTH * 2) {
        let x = (xs as f32 * 0.5 - SCREEN_WIDTH as f32 * 0.5) / ZOOM;
        let y = f(x);
        let ys = (SCREEN_HEIGHT / 2 - (y * ZOOM) as i32) as f32;

        if !ys.is_finite() {
            last_ys = None;
            continue;
        }

        if let Some(last_ys_value) = last_ys {
            if (last_ys_value - ys).abs() > ZOOM {
                last_ys = None;
                continue;
            }

            draw_circle(0.5 * xs as f32, ys, LINE_THICKNESS * 0.5, color);
            draw_line(0.5 * xs as f32 - 0.5, last_ys_value, 0.5 * xs as f32, ys, LINE_THICKNESS, color);
        }

        last_ys = Some(ys);
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut network = Network::random(
        &[1, 20, 10, 1],
        &[Tanh::new(), Tanh::new(), LeakyReLU::new(1.0)],
        &distr::Uniform::new(-0.2, 0.2).unwrap(),
    ).unwrap();

    let inputs: Vec<f32> = (0..=(SCREEN_WIDTH))
        .step_by(4)
        .map(|xs| {
            (xs as f32 - SCREEN_WIDTH as f32 * 0.5) / ZOOM
        }).filter(|&x| function_to_approximate(x).is_finite())
        .collect();

    let dataset: Vec<Sample> = inputs.into_iter()
        .map(|input| Sample::new(
            dvector![input],
            dvector![function_to_approximate(input)]
        )).collect();

    loop {
        clear_background(BACKGROUND_COLOR);

        draw_grid();
        draw_function(&function_to_approximate, LINE_COLOR);
        draw_function(&|x| {
            let outputs = network.forward(dvector![x]).unwrap();
            outputs[0]
        }, APPROXIMATION_LINE_COLOR);

        for _ in 0..200 {
            network.learn(&dataset, &MSE, 0.02).unwrap();
        }

        next_frame().await;
    }
}