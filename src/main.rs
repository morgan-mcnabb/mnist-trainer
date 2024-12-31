
use eframe::NativeOptions;
use neural_net::gui::GuiApp;
use std::panic;

fn main() {
    let native_options = NativeOptions::default();
    panic::set_hook(Box::new(|info| {
        if let Some(s) = info.payload().downcast_ref::<&str>() {
            eprintln!("Panic occurred: {}", s);
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            eprintln!("Panic occurred: {}", s);
        } else {
            eprintln!("Panic occurred: {:?}", info.payload());
        }

        if let Some(location) = info.location() {
            eprintln!(
                "Location: file '{}' at line {}, column {}",
                location.file(),
                location.line(),
                location.column()
            );
        }

    }));


    let _ =eframe::run_native(
        "Neural Network GUI",
        native_options,
        Box::new(|_cc| Ok(Box::new(GuiApp::default()))),
    );
}
