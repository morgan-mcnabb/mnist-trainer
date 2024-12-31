use eframe::egui;
use crate::config::Config;
use crate::data::loader::load_mnist;
use crate::network::initialize_network;
use crate::network::activation::Activation;
use crate::training::trainer::{train, forward_pass};
use crate::data::dataset::Sample;
use crate::metrics::accuracy::evaluate;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Serialize, Deserialize)]
pub struct AppState {
    config: Config,
    training: bool,
    progress: f32,
    train_accuracy: f32,
    test_accuracy: f32,
    status: String,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            config: Config::default(),
            training: false,
            progress: 0.0,
            train_accuracy: 0.0,
            test_accuracy: 0.0,
            status: "Idle".to_string(),
        }
    }
}
pub struct GuiApp {
    state: Arc<Mutex<AppState>>,
}

impl Default for GuiApp {
    fn default() -> Self {
        Self {
            state: Arc::new(Mutex::new(AppState::default())),
        }
    }
}

impl eframe::App for GuiApp {

    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        let mut state = self.state.lock().unwrap();

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Neural Network Trainer for MNIST");

            ui.separator();

            ui.collapsing("Configuration", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Epochs:");
                    ui.add(egui::DragValue::new(&mut state.config.epochs).range(1..=500));
                });

                ui.horizontal(|ui| {
                    ui.label("Learning Rate:");
                    ui.add(egui::DragValue::new(&mut state.config.learning_rate).clamp_range(0.0001..=1.0));
                });

                let mut layers_input = state
                    .config
                    .layers
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<String>>()
                    .join(",");

                if ui
                    .add(egui::TextEdit::singleline(&mut layers_input).hint_text("e.g. 784,512,256,64,10"))
                    .changed() {
                        state.config.layers = layers_input
                            .split(",")
                            .map(|s| s.trim().parse().unwrap_or(0))
                            .collect();
                }

                let mut activations_input = state
                    .config
                    .activations
                    .join(",");

                if ui
                    .add(egui::TextEdit::singleline(&mut activations_input).hint_text("e.g. sigmoid,relu,sigmoid"))
                        .changed() {
                            state.config.activations = activations_input
                                .split(",")
                                .map(|s| s.trim().to_lowercase())
                                .collect();
                }
            });

            ui.separator();
        });
    }
}
