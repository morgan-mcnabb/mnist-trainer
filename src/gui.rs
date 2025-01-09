use eframe::egui;
use crate::config::Config;
use crate::data::loader::load_mnist;
use crate::network::initialize_network;
use crate::network::layer::Layer;
use crate::network::activation::Activation;
use crate::training::trainer::{train, train_with_minibatch,forward_pass};
use crate::data::dataset::Sample;
use crate::metrics::accuracy::evaluate;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone, Copy)]
pub enum TrainingState {
    Idle,
    Training,
    Paused,
    Complete,
}

#[derive(Serialize, Deserialize)]
pub struct AppState {
    pub config: Config,
    pub progress: f32,
    pub train_accuracy: f32,
    pub test_accuracy: f32,
    pub status: String,
    pub network: Option<Vec<crate::network::layer::Layer>>,
    pub train_accuracy_history: Vec<f32>,
    pub test_accuracy_history: Vec<f32>,
    pub selected_sample_index: usize,
    pub prediction_result: Option<(usize, usize)>,
    pub needs_repaint: bool,
    pub training_state: TrainingState,

    #[serde(skip)]
    pub texture_cache: HashMap<usize, egui::TextureHandle>,

    #[serde(skip)]
    pub test_set: Vec<Sample>,

    #[serde(skip)]
    pub train_set: Vec<Sample>,

}

impl Default for AppState {
    fn default() -> Self {

        Self {
            config: Config::default(),
            progress: 0.0,
            train_accuracy: 0.0,
            test_accuracy: 0.0,
            status: "Idle".to_string(),
            network: None,
            train_accuracy_history: Vec::new(),
            test_accuracy_history: Vec::new(),
            selected_sample_index: 0,
            prediction_result: None,
            needs_repaint: false,
            texture_cache: std::collections::HashMap::new(),
            test_set: Vec::new(),
            train_set: Vec::new(),
            training_state: TrainingState::Idle,
        }
    }
}

pub struct GuiApp {
    state: Arc<Mutex<AppState>>,
}

impl Default for GuiApp {
    fn default() -> Self {
        let (train_set, test_set) = load_mnist();

        Self {
            state: Arc::new(Mutex::new(AppState {
                train_set,
                test_set,
                ..AppState::default()
            })),
        }
    }
}

impl GuiApp {
    fn ui_configuration(&self, ui: &mut egui::Ui) {
        let mut state = self.state.lock().unwrap();

        ui.collapsing("Configuration", |ui| {
            ui.horizontal(|ui| {
                ui.label("Epochs:");
                ui.add(egui::DragValue::new(&mut state.config.epochs).range(1..=1000));
            });

            ui.horizontal(|ui| {
                ui.label("Learning Rate:");
                ui.add(egui::DragValue::new(&mut state.config.learning_rate).range(0.0001..=1.0));
            });

            let mut layers_input = state
                .config
                .layers
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<String>>()
                .join(",");
            if ui
                .add(egui::TextEdit::singleline(&mut layers_input).hint_text("e.g., 784,256,128,64,10"))
                .changed()
            {
                state.config.layers = layers_input
                    .split(',')
                    .map(|s| s.trim().parse().unwrap_or(0))
                    .collect();
            }

            let mut activations_input = state.config.activations.join(",");
            if ui
                .add(egui::TextEdit::singleline(&mut activations_input).hint_text("e.g., sigmoid,relu,relu"))
                .changed()
            {
                state.config.activations = activations_input
                    .split(',')
                    .map(|s| s.trim().to_lowercase())
                    .collect();
            }

            if state.config.layers.len() < 2 {
                ui.colored_label(egui::Color32::RED, "Error: At least two layers required (input and output).");
            }
            if state.config.activations.len() != state.config.layers.len() - 1 {
                ui.colored_label(egui::Color32::RED, "Error: Number of activations must be one less than number of layers.");
            }
        });
    }

    fn ui_training_controls(&self, ui: &mut egui::Ui) {
        let (training_state, progress) = {
            let lock = self.state.lock().unwrap();
            (lock.training_state, lock.progress)
        };

        ui.horizontal(|ui| {
            let start_enabled = matches!(training_state, TrainingState::Idle | TrainingState::Complete);

            if ui.add_enabled(start_enabled, egui::Button::new("Start Training")).clicked() && start_enabled {
                let state_clone = Arc::clone(&self.state);
                {
                    let mut lock = state_clone.lock().unwrap();
                    lock.training_state = TrainingState::Training;
                    lock.status = "Training started".to_string();
                    lock.train_accuracy_history.clear();
                    lock.test_accuracy_history.clear();
                }
                self.spawn_training_thread(state_clone);
            }

            match training_state {
                TrainingState::Training => {
                    if ui.button("Pause Training").clicked() {
                        let mut lock = self.state.lock().unwrap();
                        lock.training_state = TrainingState::Paused;
                        lock.status = "Pausing training...".to_string();
                    }

                    if ui.button("Stop Training").clicked() {
                        let mut lock = self.state.lock().unwrap();
                        lock.training_state = TrainingState::Idle;
                        lock.status = "Stopping training...".to_string();
                    }
                }
                TrainingState::Paused => {
                    if ui.button("Resume Training").clicked() {
                        let mut lock = self.state.lock().unwrap();
                        lock.training_state = TrainingState::Training;
                        lock.status = "Resuming training...".to_string();
                    }

                    if ui.button("Stop Training").clicked() {
                        let mut lock = self.state.lock().unwrap();
                        lock.training_state = TrainingState::Idle;
                        lock.status = "Stopping training...".to_string();
                    }
                }
                _ => {
                    ui.add_enabled(false, egui::Button::new("Pause Training"));
                    ui.add_enabled(false, egui::Button::new("Stop Training"));
                }
            }

            if ui.button("Save Model").clicked() {
                let mut lock = self.state.lock().unwrap();
                if let Some(ref network) = lock.network {
                    let serialized = serde_json::to_string(network).unwrap();
                    std::fs::write("trained_model.json", serialized).expect("Unable to save model");
                    lock.status = "Model saved successfully.".to_string();
                } else {
                    lock.status = "No trained network to save.".to_string();
                }
            }

            if ui.button("Load Model").clicked() {
                let mut lock = self.state.lock().unwrap();
                let data = std::fs::read_to_string("trained_model.json");
                match data {
                    Ok(content) => {
                        let network: Vec<Layer> = serde_json::from_str(&content).unwrap();
                        lock.network = Some(network);
                        lock.status = "Model loaded successfully.".to_string();
                    }
                    Err(_) => {
                        lock.status = "Failed to load model.".to_string();
                    }
                }
            }
        });
    }

    fn ui_status(&self, ui: &mut egui::Ui, status: &str, training_state: TrainingState) {
        let status_color = match training_state {
            TrainingState::Idle => egui::Color32::BLUE,
            TrainingState::Training => egui::Color32::GOLD,
            TrainingState::Paused => egui::Color32::ORANGE,
            TrainingState::Complete => egui::Color32::GREEN,
        };

        ui.horizontal(|ui| {
            ui.label("Status:");
            ui.label(
                egui::RichText::new(status)
                    .color(status_color)
                    .strong(),
            );
        });
    }

    fn ui_progress_and_accuracy(&self, ui: &mut egui::Ui) {
        let (progress, train_acc, test_acc) = {
            let lock = self.state.lock().unwrap();
            (lock.progress, lock.train_accuracy, lock.test_accuracy)
        };

        ui.add(egui::ProgressBar::new(progress / 100.0).show_percentage());
        ui.separator();

        ui.horizontal(|ui| {
            ui.label(format!("Training Accuracy: {:.2}%", train_acc));
            ui.label(format!("Testing Accuracy: {:.2}%", test_acc));
        });
    }

    fn ui_training_metrics(&self, ui: &mut egui::Ui) {
        let (train_history, test_history) = {
            let lock = self.state.lock().unwrap();
            (
                lock.train_accuracy_history.clone(),
                lock.test_accuracy_history.clone(),
            )
        };

        ui.collapsing("Training Metrics", |ui| {
            egui_plot::Plot::new("Accuracy Plot")
                .view_aspect(2.0)
                .show(ui, |plot_ui| {
                    let train_data: Vec<[f64; 2]> = train_history
                        .iter()
                        .enumerate()
                        .map(|(i, &acc)| [i as f64, acc as f64])
                        .collect();

                    let test_data: Vec<[f64; 2]> = test_history
                        .iter()
                        .enumerate()
                        .map(|(i, &acc)| [i as f64, acc as f64])
                        .collect();

                    plot_ui.line(
                        egui_plot::Line::new(egui_plot::PlotPoints::from_iter(train_data))
                            .name("Train Accuracy"),
                    );
                    plot_ui.line(
                        egui_plot::Line::new(egui_plot::PlotPoints::from_iter(test_data))
                            .name("Test Accuracy"),
                    );
                });
        });
    }

    fn ui_prediction(&self, ui: &mut egui::Ui) {
        ui.collapsing("Make a Prediction", |ui| {
            let (network_exists, selected_index) = {
                let lock = self.state.lock().unwrap();
                (lock.network.is_some(), lock.selected_sample_index)
            };

            if network_exists {
                let test_set = {
                    let lock = self.state.lock().unwrap();
                    lock.test_set.clone()
                };
                if !test_set.is_empty() {
                    ui.horizontal(|ui| {
                        ui.label("Select Test Sample Index:");
                        let mut lock = self.state.lock().unwrap();
                        ui.add(egui::DragValue::new(&mut lock.selected_sample_index).range(0..=test_set.len()-1));
                    });

                    let new_selected_index = {
                        let lock = self.state.lock().unwrap();
                        lock.selected_sample_index
                    };

                    if new_selected_index < test_set.len() {
                        let sample = &test_set[new_selected_index];
                        let texture_id = {
                            let mut lock = self.state.lock().unwrap();
                            if let Some(texture) = lock.texture_cache.get(&new_selected_index) {
                                texture.clone()
                            } else {
                                let image = convert_to_image(&sample.inputs);
                                let texture = ui.ctx().load_texture(
                                    format!("sample_image_{}", new_selected_index),
                                    egui::ColorImage::from_rgb([28, 28], &image),
                                    egui::TextureOptions::NEAREST,
                                );

                                lock.texture_cache.insert(new_selected_index, texture.clone());
                                texture
                            }
                        };

                        ui.image(&texture_id);

                        if ui.button("Predict").clicked() {
                            let predicted_label = {
                                let lock = self.state.lock().unwrap();
                                let network = lock.network.as_ref().unwrap();
                                predict(network, sample)
                            };

                            let actual_label = sample
                                .target
                                .iter()
                                .position(|&v| v == 1.0)
                                .unwrap_or(0);

                            {
                                let mut lock = self.state.lock().unwrap();
                                lock.prediction_result = Some((predicted_label, actual_label));
                            }
                        }

                        let prediction = {
                            let lock = self.state.lock().unwrap();
                            lock.prediction_result
                        };
                        if let Some((prediction_result, actual)) = prediction {
                            ui.label(format!("Prediction: {}", prediction_result));
                            ui.label(format!("Actual Label: {}", actual));
                        }
                    } else {
                        ui.label("Invalid sample index.");
                    }
                } else {
                    ui.label("No test samples available.");
                }
            } else {
                ui.label("Train the network first to make predictions.");
            }
        });
    }

    fn ui_logs(&self, ui: &mut egui::Ui) {
        let mut lock = self.state.lock().unwrap();
        ui.collapsing("Logs", |ui| {
            ui.text_edit_multiline(&mut lock.status);
        });
    }

    fn spawn_training_thread(&self, state_clone: Arc<Mutex<AppState>>) {
        thread::spawn(move || {
            let (config, train_set, test_set) = {
                let lock = state_clone.lock().unwrap();
                (
                    lock.config.clone(),
                    lock.train_set.clone(),
                    lock.test_set.clone(),
                )
            };

            let activations = config
                .activations
                .iter()
                .map(|s| match s.as_str() {
                    "sigmoid" => Activation::Sigmoid,
                    "relu" => Activation::ReLU,
                    "softmax" => Activation::Softmax,
                    _ => Activation::Sigmoid,
                })
                .collect::<Vec<_>>();

            let mut network = initialize_network(&config.layers, &activations);

            for epoch in 0..config.epochs {
                {
                    let mut lock = state_clone.lock().unwrap();
                    match lock.training_state {
                        TrainingState::Paused => {
                            lock.status = "Training paused.".to_string();
                            lock.needs_repaint = true;
                        }
                        TrainingState::Idle => {
                            lock.status = "Training halted.".to_string();
                            lock.needs_repaint = true;
                            return;
                        }
                        TrainingState::Complete => {
                            lock.status = "Training completed.".to_string();
                            lock.needs_repaint = true;
                            return;
                        }
                        _ => {}
                    }
                }

                loop {
                    {
                        let mut lock = state_clone.lock().unwrap();
                        match lock.training_state {
                            TrainingState::Training => {
                                lock.status = format!("Training... Epoch {}/{}", epoch + 1, config.epochs);
                                break;
                            }
                            TrainingState::Complete => {
                                lock.status = "Training completed.".to_string();
                                lock.needs_repaint = true;
                                return;
                            }
                            TrainingState::Idle => {
                                lock.status = "Training halted.".to_string();
                                lock.needs_repaint = true;
                                return;
                            }
                            _ => {}
                        }
                    }
                    thread::sleep(Duration::from_millis(100));
                }

                //train(&mut network, &train_set, 1, config.learning_rate);
                train_with_minibatch(&mut network, &train_set, 1, config.learning_rate, config.batch_size,);

                {
                    let mut lock = state_clone.lock().unwrap();
                    lock.train_accuracy = evaluate(&mut network, &train_set);
                    lock.test_accuracy = evaluate(&mut network, &test_set);
                    lock.network = Some(network.clone());
                    lock.needs_repaint = true;
                }

                {
                    let mut lock = state_clone.lock().unwrap();
                    lock.status = format!("Training... Epoch {}/{}", epoch + 1, config.epochs);
                    lock.progress = ((epoch + 1) as f32 / config.epochs as f32) * 100.0;
                }

                thread::sleep(Duration::from_millis(10));
            }

            {
                let mut lock = state_clone.lock().unwrap();
                lock.progress = 100.0;
                lock.status = "Training complete".to_string();
                lock.training_state = TrainingState::Complete;
                lock.network = Some(network);
                lock.needs_repaint = true;
            }
        });
    }}

impl eframe::App for GuiApp {

    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
            let (training_state, needs_repaint, status) = {
            let lock = self.state.lock().unwrap();
            (
                lock.training_state,
                lock.needs_repaint,
                lock.status.clone(),
            )
        };

        if needs_repaint {
            let mut lock = self.state.lock().unwrap();
            ctx.request_repaint();
            lock.needs_repaint = false;
        }

        if training_state == TrainingState::Training {
            ctx.request_repaint_after(Duration::from_millis(100));
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Neural Network Trainer for MNIST");
            ui.separator();

            self.ui_configuration(ui);

            ui.separator();

            self.ui_training_controls(ui);

            self.ui_status(ui, &status, training_state);

            self.ui_progress_and_accuracy(ui);

            ui.separator();

            self.ui_training_metrics(ui);

            ui.separator();

            self.ui_prediction(ui);

            self.ui_logs(ui);
        });
    }
}

fn predict(layers: &[Layer], sample: &Sample) -> usize {
    let mut layers = layers.to_vec(); 
    forward_pass(&mut layers, &sample.inputs);
    let output_index = layers.len() - 1;
    layers[output_index]
        .neurons
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.activated_value.partial_cmp(&b.1.activated_value).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn convert_to_image(inputs: &ndarray::Array1<f32>) -> Vec<u8> {
    // was doing [pixel, pixel] instead of [pixel, pixel, pixel]
    // hours wasted: 4
       inputs.iter().flat_map(|&v| {
           let pixel = (v * 255.0) as u8;
           [pixel, pixel, pixel]
       }).collect()
}
