use eframe::egui;
use crate::config::Config;
use crate::data::loader::load_mnist;
use crate::network::initialize_network;
use crate::network::activation::Activation;
use crate::training::trainer::train;
use crate::data::dataset::Sample;
use crate::metrics::accuracy::evaluate;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Serialize, Deserialize)]
pub struct AppState {
    pub config: Config,
    pub training: bool,
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

        // forgot to drop this lock. hours wasted: 8
        let (training, needs_repaint, status) = {
            let state_lock = self.state.lock().unwrap();
            (state_lock.training, state_lock.needs_repaint, state_lock.status.clone())
        };

        if needs_repaint {
            let mut state_lock =  self.state.lock().unwrap();
            ctx.request_repaint();
            state_lock.needs_repaint = false;
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Neural Network Trainer for MNIST");

            ui.separator();

            ui.collapsing("Configuration", |ui| {

                {
                    let mut state_lock = self.state.lock().unwrap();
                    ui.horizontal(|ui| {
                        ui.label("Epochs:");
                        ui.add(egui::DragValue::new(&mut state_lock.config.epochs).range(1..=1000));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Learning Rate:");
                        ui.add(egui::DragValue::new(&mut state_lock.config.learning_rate).range(0.0001..=1.0));
                    });

                    let mut layers_input = state_lock
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
                        state_lock.config.layers = layers_input
                            .split(',')
                            .map(|s| s.trim().parse().unwrap_or(0))
                            .collect();
                    }

                    let mut activations_input = state_lock
                        .config
                        .activations
                        .join(",");
                    if ui
                        .add(egui::TextEdit::singleline(&mut activations_input).hint_text("e.g., sigmoid,relu,relu"))
                        .changed()
                    {
                        state_lock.config.activations = activations_input
                            .split(',')
                            .map(|s| s.trim().to_lowercase())
                            .collect();
                    }

                    if state_lock.config.layers.len() < 2 {
                        ui.colored_label(egui::Color32::RED, "Error: At least two layers required (input and output).");
                    }
                    if state_lock.config.activations.len() != state_lock.config.layers.len() - 1 {
                        ui.colored_label(egui::Color32::RED, "Error: Number of activations must be one less than number of layers.");
                    }
                } // lock is dropped here
            });

            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("Start Training").clicked() && !training {
                    
                    let state_clone = Arc::clone(&self.state);
                    {
                        let mut state_lock = state_clone.lock().unwrap();
                        state_lock.training = true;
                        state_lock.status = "Training started".to_string();
                        state_lock.train_accuracy_history.clear();
                        state_lock.test_accuracy_history.clear();
                    }

                    thread::spawn(move || {
                        { 
                            println!("DEBUG: Training thread started. Attempting to lock state_clone...");
                            let mut state_lock = state_clone.lock().unwrap();
                            let config = state_lock.config.clone();
                            drop(state_lock);


                            println!("DEBUG: Successfully locked state. Cloning config...");
                            println!("DEBUG: Dropped lock, now loading MNIST...");

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
                                    let mut state_lock  = state_clone.lock().unwrap();
                                    state_lock.status = format!("Training... Epoch {}/{}", epoch + 1, config.epochs);
                                    state_lock.progress = (epoch as f32 / config.epochs as f32) * 100.0;
                                }

                                train(&mut network, &train_set, 1, config.learning_rate, &test_set);

                                {
                                    let mut state_lock = state_clone.lock().unwrap();
                                    state_lock.train_accuracy = evaluate(&mut network, &train_set);
                                    state_lock.test_accuracy = evaluate(&mut network, &test_set);
                                    let train_acc = state_lock.train_accuracy;
                                    let test_acc = state_lock.test_accuracy;
                                    state_lock.train_accuracy_history.push(train_acc);
                                    state_lock.test_accuracy_history.push(test_acc);
                                    state_lock.network = Some(network.clone()); 
                                    state_lock.needs_repaint = true;
                                }

                                std::thread::sleep(std::time::Duration::from_millis(10));
                            }

                            {
                                let mut state_lock = state_clone.lock().unwrap();
                                state_lock.progress = 100.0;
                                state_lock.status = "Training complete".to_string();
                                state_lock.training = false;
                                state_lock.network = Some(network);
                                state_lock.needs_repaint = true;
                            }
                        }
                    });
                }

                if ui.button("Save Model").clicked() {
                    let mut state_lock = self.state.lock().unwrap();
                    if let Some(ref network) = state_lock.network {
                        let serialized = serde_json::to_string(network).unwrap();
                        std::fs::write("trained_model.json", serialized).expect("Unable to save model");
                        state_lock.status = "Model saved successfully.".to_string();
                    } else {
                        state_lock.status = "No trained network to save.".to_string();
                    }
                }

                if ui.button("Load Model").clicked() {
                    let mut state_lock = self.state.lock().unwrap();
                    let data = std::fs::read_to_string("trained_model.json");
                    match data {
                        Ok(content) => {
                            let network: Vec<crate::network::layer::Layer> = serde_json::from_str(&content).unwrap();
                            state_lock.network = Some(network);
                            state_lock.status = "Model loaded successfully.".to_string();
                        }
                        Err(_) => {
                            state_lock.status = "Failed to load model.".to_string();
                        }
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label("Status:");
                ui.label(&status);
            });

            let progress = {
                let state_lock = self.state.lock().unwrap();
                state_lock.progress
            };

            ui.add(egui::ProgressBar::new(progress / 100.0).show_percentage());

            let (train_acc, test_acc) = {
                let state_lock = self.state.lock().unwrap();
                (state_lock.train_accuracy, state_lock.test_accuracy)
            };

            ui.separator();
            ui.horizontal(|ui| {
                ui.label(format!("Training Accuracy: {:.2}%", train_acc));
                ui.label(format!("Testing Accuracy: {:.2}%", test_acc));
            });

           ui.collapsing("Training Metrics", |ui| {
                let (train_history, test_history) = {
                    let state_lock = self.state.lock().unwrap();
                    (state_lock.train_accuracy_history.clone(), state_lock.test_accuracy_history.clone())
                };

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
            ui.separator();

            
            ui.collapsing("Make a Prediction", |ui| {
                println!("Make a prediction clicked");
                let (network_exists, selected_index) = {
                    let state_lock = self.state.lock().unwrap();
                    (state_lock.network.is_some(), state_lock.selected_sample_index)
                };

                println!("before network exists if statement");
                if network_exists {
                    println!("loading mnist");
                    let test_set = load_mnist().1;
                    if !test_set.is_empty() {
                        ui.horizontal(|ui| {
                            ui.label("Select Test Sample Index:");
                            let mut state_lock = self.state.lock().unwrap();
                            ui.add(egui::DragValue::new(&mut state_lock.selected_sample_index).range(0..=test_set.len()-1));
                        });

                        let new_selected_index = {
                            let state_lock = self.state.lock().unwrap();
                            state_lock.selected_sample_index
                        };

                        if new_selected_index < test_set.len() {
                            let sample = &test_set[new_selected_index];
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

                                let predicted_label = {
                                    let state_lock = self.state.lock().unwrap();
                                    let network = state_lock.network.as_ref().unwrap();
                                    predict(network, sample)
                                };

                                let actual_label = sample
                                    .target
                                    .iter()
                                    .position(|&v| v == 1.0)
                                    .unwrap_or(0);

                                {
                                    let mut state_lock = self.state.lock().unwrap();
                                    state_lock.prediction_result = Some((predicted_label, actual_label));
                                }
                            }

                            let pred_res = {
                                let state_lock = self.state.lock().unwrap();
                                state_lock.prediction_result
                            };
                            if let Some((pred, actual)) = pred_res {
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
                let mut state_lock = self.state.lock().unwrap();
                ui.text_edit_multiline(&mut state_lock.status);
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
