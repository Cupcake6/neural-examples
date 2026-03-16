use macroquad::prelude::*;
use neural::prelude::*;
use nalgebra::dvector;

const BUFFER_ROWS: usize = 120;
const BUFFER_COLUMNS: usize = 160;

fn window_conf() -> Conf {
    Conf {
        window_title: "Dot Classification".to_string(),
        window_width: 800,
        window_height: 600,
        ..Default::default()
    }
}

fn draw_buffer(buffer: &[Color], rows: usize, columns: usize) {
    let w = screen_width() / columns as f32;
    let h = screen_height() / rows as f32;

    for row in 0..rows {
        for col in 0..columns {
            let i = col + row * columns;
            if let Some(&color) = buffer.get(i) {
                draw_rectangle(col as f32 * w, row as f32 * h, w, h, color);
            }
        }
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut buffer = [BLACK; BUFFER_ROWS * BUFFER_COLUMNS];

    let mut network = Network::random(
        &[2, 8, 8, 1],
        &[Swish::new(), Tanh::new(), Sigmoid::new()],
        &distr::Uniform::new(-0.2, 0.2).unwrap(),
    )
    .unwrap();

    let mut dataset = Vec::<Sample>::new();

    loop {
        let (mx, my) = mouse_position();

        let mx = mx / screen_width() * 2.0 - 1.0;
        let my = my / screen_height() * -2.0 + 1.0;

        if is_mouse_button_pressed(MouseButton::Left) {
            dataset.push(Sample::new(dvector![mx, my], dvector![1.0]));
        }

        if is_mouse_button_pressed(MouseButton::Right) {
            dataset.push(Sample::new(dvector![mx, my], dvector![0.0]));
        }

        for _ in 0..1000 {
            network.learn(&dataset, &BCE, 0.005).unwrap();
        }

        for row in 0..BUFFER_ROWS {
            for col in 0..BUFFER_COLUMNS {
                let i = col + row * BUFFER_COLUMNS;
                let Some(pixel) = buffer.get_mut(i) else { continue };

                let x = col as f32 / BUFFER_COLUMNS as f32 * 2.0 - 1.0;
                let y = row as f32 / BUFFER_ROWS as f32 * -2.0 + 1.0;

                let output = network.forward(dvector![x, y]).unwrap()[0];

                *pixel = Color::new(output * 0.7, 0.3, (1.0 - output) * 0.7, 1.0);
            }
        }

        draw_buffer(&buffer, BUFFER_ROWS, BUFFER_COLUMNS);

        for sample in &dataset {
            let pos = sample.inputs();
            let x = (pos[0] + 1.0) / 2.0 * screen_width();
            let y = (pos[1] - 1.0) / -2.0 * screen_height();

            draw_circle(
                x,
                y,
                5.0,
                if sample.expected_outputs()[0] == 1.0 { RED } else { BLUE },
            );
        }

        next_frame().await;
    }
}