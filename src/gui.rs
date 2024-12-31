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
use egui_plot::Plot;

#[derive(Serialize, Deserialize)]
pub struct AppState {
    config: Config,
    training: bool,
    progress: f32,
    train_accuracy: f32,
    test_accuracy: f32,
    status: String,
    network: Option<Vec<crate::network::layer::Layer>>,
    train_accuracy_history: Vec<f32>,
    test_accuracy_history: Vec<f32>,
    selected_sample_index: usize,
    prediction_result: Option<(usize, usize)>,
    needs_repaint: bool,
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
            network: None,
            train_accuracy_history: Vec::new(),
            test_accuracy_history: Vec::new(),
            selected_sample_index: 0,
            prediction_result: None,
            needs_repaint: false,
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

        if state.needs_repaint {
            ctx.request_repaint();
            state.needs_repaint = false;
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Neural Network Trainer for MNIST");

            ui.separator();

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

                let mut activations_input = state
                    .config
                    .activations
                    .join(",");
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

            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("Start Training").clicked() && !state.training {
                    let state_clone = Arc::clone(&self.state);
                    state_clone.lock().unwrap().training = true;
                    state_clone.lock().unwrap().status = "Training started".to_string();
                    state_clone.lock().unwrap().train_accuracy_history.clear();
                    state_clone.lock().unwrap().test_accuracy_history.clear();

                    thread::spawn(move || {
                        let mut state = state_clone.lock().unwrap();
                        let config = state.config.clone();
                        drop(state); // release the lock first. stupid. hours wasted: 2.5

                        let (train_set, test_set) = load_mnist();
                        
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
                                let mut state = state_clone.lock().unwrap();
                                state.status = format!("Training... Epoch {}/{}", epoch + 1, config.epochs);
                                state.progress = (epoch as f32 / config.epochs as f32) * 100.0;
                            }

                            train(&mut network, &train_set, 1, config.learning_rate, &test_set);

                            {
                                let mut state = state_clone.lock().unwrap();
                                state.train_accuracy = evaluate(&mut network, &train_set);
                                state.test_accuracy = evaluate(&mut network, &test_set);
                                let train_acc = state.train_accuracy;
                                let test_acc = state.test_accuracy;
                                state.train_accuracy_history.push(train_acc);
                                state.test_accuracy_history.push(test_acc);
                                state.network = Some(network.clone()); 
                                state.needs_repaint = true;
                            }
                        }

                        {
                            let mut state = state_clone.lock().unwrap();
                            state.progress = 100.0;
                            state.status = "Training complete".to_string();
                            state.training = false;
                            state.network = Some(network);
                            state.needs_repaint = true;
                        }
                    });
                }

                if ui.button("Save Model").clicked() {
                    if let Some(ref network) = state.network {
                        let serialized = serde_json::to_string(network).unwrap();
                        std::fs::write("trained_model.json", serialized).expect("Unable to save model");
                        state.status = "Model saved successfully.".to_string();
                    } else {
                        state.status = "No trained network to save.".to_string();
                    }
                }

                if ui.button("Load Model").clicked() {
                    let data = std::fs::read_to_string("trained_model.json");
                    match data {
                        Ok(content) => {
                            let network: Vec<crate::network::layer::Layer> = serde_json::from_str(&content).unwrap();
                            state.network = Some(network);
                            state.status = "Model loaded successfully.".to_string();
                        }
                        Err(_) => {
                            state.status = "Failed to load model.".to_string();
                        }
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label("Status:");
                ui.label(&state.status);
            });

            ui.add(egui::ProgressBar::new(state.progress / 100.0).show_percentage());

            ui.separator();
            ui.horizontal(|ui| {
                ui.label(format!("Training Accuracy: {:.2}%", state.train_accuracy));
                ui.label(format!("Testing Accuracy: {:.2}%", state.test_accuracy));
            });

            ui.collapsing("Training Metrics", |ui| {
                egui_plot::Plot::new("Accuracy Plot")
                    .view_aspect(2.0)
                    .show(ui, |plot_ui| {
                        let train_data: Vec<[f64; 2]> = state.train_accuracy_history.iter().enumerate()
                            .map(|(i, &acc)| [i as f64, acc as f64])
                            .collect();
                        let test_data: Vec<[f64; 2]> = state.test_accuracy_history.iter().enumerate()
                            .map(|(i, &acc)| [i as f64, acc as f64])
                            .collect();

                        plot_ui.line(egui_plot::Line::new(egui_plot::PlotPoints::from_iter(train_data)).name("Train Accuracy"));
                        plot_ui.line(egui_plot::Line::new(egui_plot::PlotPoints::from_iter(test_data)).name("Test Accuracy"));
                    });
            });

            ui.separator();

            ui.collapsing("Make a Prediction", |ui| {
                if state.network.is_some() {
                    let test_set = load_mnist().1;
                    if !test_set.is_empty() {
                        ui.horizontal(|ui| {
                            ui.label("Select Test Sample Index:");
                            ui.add(egui::DragValue::new(&mut state.selected_sample_index).range(0..=test_set.len()-1));
                        });

                        if state.selected_sample_index < test_set.len() {
                            let sample = &test_set[state.selected_sample_index];
                            let image = convert_to_image(&sample.inputs);
                            let texture_id = ui.ctx().load_texture(
                                "sample_image",
                                egui::ColorImage::from_rgb([28, 28], &image),
                                egui::TextureOptions {
                                    magnification: egui::TextureFilter::Nearest,
                                    minification: egui::TextureFilter::Nearest,
                                    wrap_mode: egui::TextureWrapMode::ClampToEdge,
                                    mipmap_mode: None,
                                },
                            );

                            ui.image(&texture_id); 

                            if ui.button("Predict").clicked() {
                                let network = state.network.as_ref().unwrap();
                                let predicted_label = predict(network, sample);
                                let actual_label = sample
                                    .target
                                    .iter()
                                    .position(|&v| v == 1.0)
                                    .unwrap_or(0);
                                state.prediction_result = Some((predicted_label, actual_label));
                            }

                            if let Some((pred, actual)) = state.prediction_result {
                                ui.label(format!("Prediction: {}", pred));
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

            ui.collapsing("Logs", |ui| {
                ui.text_edit_multiline(&mut state.status);
            });
        });

    }}

fn predict(layers: &[crate::network::layer::Layer], sample: &Sample) -> usize {
    let mut layers = layers.to_vec(); 
    crate::training::trainer::forward_pass(&mut layers, &sample.inputs);
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
    inputs.iter().map(|&v| (v * 255.0) as u8).collect()
}
