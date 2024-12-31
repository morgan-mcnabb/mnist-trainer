
use eframe::NativeOptions;
use neural_net::gui::GuiApp;
use std::panic;

fn main() {
    let native_options = NativeOptions::default();
    panic::set_hook(Box::new(|info| {
        // Print or log the panic info
        println!("Panic occurred: {:?}", info);
    }));
    let _ =eframe::run_native(
        "Neural Network GUI",
        native_options,
        Box::new(|_cc| Ok(Box::new(GuiApp::default()))),
    );
}
