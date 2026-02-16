import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

class SquirrelTrajectoryTracker:
    def __init__(self, video_paths, condition_name):
        """
        video_paths: Liste von Video-Pfaden
        condition_name: z.B. 'leaf', 'cups', 'disco'
        """
        self.video_paths = video_paths
        self.condition_name = condition_name
        self.trajectories = []
        
    def track_single_video(self, video_path, 
                          crop_percent_top=0.0, 
                          crop_percent_bottom=0.0,
                          scale_percent=60,
                          min_area=500,
                          show_processing=False):
        
        # Extrahiert Trajektorie aus einem Video mit Background Subtraction
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video {video_path}")
        
        # Kernel für morphologische Operationen
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Background Subtractor
        fgbg = cv2.createBackgroundSubtractorKNN(history=500, detectShadows=True)
        
        # Listen für Trajektorie
        x_coords = []
        y_coords = []
        frame_numbers = []
        areas = []  # Größe der Detection
        
        frame_idx = 0
        
        print(f"Processing: {video_path.name}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Bildhöhe und -breite
            height, width = frame.shape[:2]
            
            # Cropping
            top_crop = int(height * crop_percent_top)
            bottom_crop = int(height * (1 - crop_percent_bottom))
            cropped_frame = frame[top_crop:bottom_crop, :]
            
            # Resize
            width_resized = int(cropped_frame.shape[1] * scale_percent / 100)
            height_resized = int(cropped_frame.shape[0] * scale_percent / 100)
            cropped_frame_resized = cv2.resize(cropped_frame, (width_resized, height_resized))
            
            # Background Subtraction
            fgmask = fgbg.apply(cropped_frame_resized)
            
            # Morphologische Operationen
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            
            # Konturen finden
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Größte Kontur = Eichhörnchen
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > min_area:
                    # Schwerpunkt berechnen
                    M = cv2.moments(largest_contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        
                        x_coords.append(cx)
                        y_coords.append(cy)
                        frame_numbers.append(frame_idx)
                        areas.append(area)
                        
                        # Optional: Visualisierung während Processing
                        if show_processing:
                            # Zentrum einzeichnen
                            cv2.circle(cropped_frame_resized, (cx, cy), 5, (0, 255, 0), -1)
                            # Kontur einzeichnen
                            cv2.drawContours(cropped_frame_resized, [largest_contour], -1, (0, 255, 0), 2)
            
            # Optional: Anzeigen während Verarbeitung
            if show_processing:
                fgmask_display = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
                combined = cv2.hconcat([cropped_frame_resized, fgmask_display])
                cv2.imshow('Processing...', combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_idx += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"  → Tracked {len(x_coords)} points over {frame_idx} frames")
        
        return {
            'x': np.array(x_coords),
            'y': np.array(y_coords),
            'frames': np.array(frame_numbers),
            'areas': np.array(areas),
            'video_name': video_path.name,
            'total_frames': frame_idx
        }
    
    def smooth_trajectory(self, trajectory, window_size=5):
        # Glättet die Trajektorie mit Savitzky-Golay Filter
        if len(trajectory['x']) < window_size:
            return trajectory
        
        # Nur smoothen wenn genug Datenpunkte
        if len(trajectory['x']) > window_size:
            trajectory['x'] = savgol_filter(trajectory['x'], window_size, 3)
            trajectory['y'] = savgol_filter(trajectory['y'], window_size, 3)
        
        return trajectory
    
    def process_all_videos(self, smooth=True, show_processing=False, **kwargs):
        # Verarbeitet alle Videos
        print(f"\n{'='*60}")
        print(f"Processing {len(self.video_paths)} videos for condition: {self.condition_name.upper()}")
        print(f"{'='*60}\n")
        
        for i, video_path in enumerate(self.video_paths):
            print(f"[{i+1}/{len(self.video_paths)}]")
            
            try:
                traj = self.track_single_video(video_path, show_processing=show_processing, **kwargs)
                
                if smooth and len(traj['x']) > 5:
                    traj = self.smooth_trajectory(traj)
                
                self.trajectories.append(traj)
                
            except Exception as e:
                print(f"  ✗ Error processing {video_path.name}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Successfully processed {len(self.trajectories)} videos")
        print(f"{'='*60}\n")
        
        return self.trajectories
    
    def plot_overlayed_trajectories(self, save_path=None, figsize=(16, 12)):
        # Überlagerte Trajektorien plotten
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.trajectories)))
        
        for i, traj in enumerate(self.trajectories):
            # Trajektorie plotten
            ax.plot(traj['x'], traj['y'], 
                    alpha=0.7,
                    linewidth=2.5,
                    color=colors[i],
                    label=traj['video_name'])
            
            # Startpunkt (grüner Kreis)
            ax.scatter(traj['x'][0], traj['y'][0], 
                    s=200, marker='o', 
                    color='green', 
                    edgecolor='black', 
                    linewidth=2, 
                    zorder=10,
                    alpha=0.8)
            
            # Endpunkt (rotes X)
            ax.scatter(traj['x'][-1], traj['y'][-1], 
                    s=200, marker='X', 
                    color='red', 
                    edgecolor='black', 
                    linewidth=2, 
                    zorder=10,
                    alpha=0.8)
        
        ax.set_xlabel('X Position (pixels)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y Position (pixels)', fontsize=13, fontweight='bold')
        ax.set_title(f'Squirrel Trajectories - {self.condition_name.upper()} Condition\n' + 
                    f'(Green = Start, Red = End)', 
                    fontsize=15, fontweight='bold', pad=15)
        
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Bessere Layout-Anpassung
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
            print(f"✓ Saved trajectory plot to: {save_path}")
        
    plt.show()
    
    def plot_with_heatmap(self, bins=50, save_path=None, figsize=(22, 10)):
        # Trajektorien + Heatmap Side-by-Side
        fig = plt.figure(figsize=figsize)
        
        # GridSpec für bessere Kontrolle
        gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.25, 
                            left=0.08, right=0.95, top=0.92, bottom=0.08)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Links: Individuelle Trajektorien
        for i, traj in enumerate(self.trajectories):
            ax1.plot(traj['x'], traj['y'], 
                    alpha=0.6, 
                    linewidth=2)
            
            # Start/End markers
            ax1.scatter(traj['x'][0], traj['y'][0], s=100, color='green', 
                    edgecolor='black', zorder=10, alpha=0.7)
            ax1.scatter(traj['x'][-1], traj['y'][-1], s=100, color='red', 
                    marker='X', edgecolor='black', zorder=10, alpha=0.7)
        
        ax1.set_title('Individual Paths', fontsize=14, fontweight='bold', pad=12)
        ax1.set_xlabel('X Position (pixels)', fontsize=12)
        ax1.set_ylabel('Y Position (pixels)', fontsize=12)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Rechts: Heatmap
        all_x = np.concatenate([t['x'] for t in self.trajectories])
        all_y = np.concatenate([t['y'] for t in self.trajectories])
        
        # Bestimme Grenzen
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        
        # Histogram2D
        heatmap, xedges, yedges = np.histogram2d(
            all_x, all_y,
            bins=bins,
            range=[[x_min, x_max], [y_min, y_max]]
        )
        
        # Glätten
        heatmap_smooth = gaussian_filter(heatmap, sigma=2)
        
        # Plot
        im = ax2.imshow(heatmap_smooth.T,
                        origin='lower',
                        extent=[x_min, x_max, y_min, y_max],
                        cmap='hot',
                        aspect='auto',
                        interpolation='bilinear')
        
        ax2.set_title('Occupancy Heatmap', fontsize=14, fontweight='bold', pad=12)
        ax2.set_xlabel('X Position (pixels)', fontsize=12)
        ax2.set_ylabel('Y Position (pixels)', fontsize=12)
        
        # Colorbar mit besserer Platzierung
        cbar = fig.colorbar(im, ax=ax2, label='Time spent (frames)', 
                        fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        
        fig.suptitle(f'Trajectory Analysis: {self.condition_name.upper()} Condition', 
                    fontsize=16, fontweight='bold', y=0.96)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
            print(f"✓ Saved heatmap plot to: {save_path}")
    
    plt.show()
    
    def print_statistics(self):
        # Detaillierte Statistiken ausgeben
        print(f"\n{'='*70}")
        print(f"TRAJECTORY STATISTICS: {self.condition_name.upper()} CONDITION")
        print(f"{'='*70}")
        print(f"Number of videos analyzed: {len(self.trajectories)}\n")
        
        path_lengths = []
        track_percentages = []
        
        for traj in self.trajectories:
            # Pfadlänge berechnen
            diffs = np.sqrt(np.diff(traj['x'])**2 + np.diff(traj['y'])**2)
            path_length = np.sum(diffs)
            path_lengths.append(path_length)
            
            # Wie viel % des Videos wurde getrackt?
            track_pct = (len(traj['x']) / traj['total_frames']) * 100
            track_percentages.append(track_pct)
            
            # Displacement (Luftlinie Start -> Ende)
            displacement = np.sqrt((traj['x'][-1] - traj['x'][0])**2 + 
                                  (traj['y'][-1] - traj['y'][0])**2)
            
            # Sinuosity (wie kurvig ist der Pfad?)
            sinuosity = path_length / (displacement + 1e-6)
            
            print(f"{traj['video_name']}:")
            print(f"  Tracked frames:     {len(traj['x'])} / {traj['total_frames']} ({track_pct:.1f}%)")
            print(f"  Path length:        {path_length:.1f} pixels")
            print(f"  Net displacement:   {displacement:.1f} pixels")
            print(f"  Sinuosity:          {sinuosity:.2f} (1=straight line)")
            print(f"  Start position:     ({traj['x'][0]:.0f}, {traj['y'][0]:.0f})")
            print(f"  End position:       ({traj['x'][-1]:.0f}, {traj['y'][-1]:.0f})")
            print(f"  Avg. blob size:     {np.mean(traj['areas']):.0f} pixels²")
            print()
        
        print(f"{'-'*70}")
        print(f"SUMMARY STATISTICS:")
        print(f"  Average path length:      {np.mean(path_lengths):.1f} ± {np.std(path_lengths):.1f} pixels")
        print(f"  Average tracking rate:    {np.mean(track_percentages):.1f}%")
        print(f"{'='*70}\n")
    
    def save_trajectories(self, output_path):
        # Trajektorien als NPZ speichern für spätere Analyse
        np.savez(output_path, 
                 trajectories=self.trajectories,
                 condition=self.condition_name)
        print(f"✓ Saved trajectories to: {output_path}")


# code von oben nutzen

if __name__ == "__main__":
    
    # Video-Pfade definieren
    video_folder = Path("C:\\Users\\hemin\\sciebo\\Master_Geoinformatik\\GI_Master_1\\Squirrels\\mp4_snippets\\leaf")
    leaf_videos = sorted(list(video_folder.glob("*.mp4")))[:5]  # Erste 5 Videos
    
    print(f"Found {len(leaf_videos)} videos")
    
    # Tracker erstellen
    tracker = SquirrelTrajectoryTracker(
        video_paths=leaf_videos,
        condition_name='leaf'
    )
    
    # Videos verarbeiten
    # Parameter anpassen je nach Videos:
    # - crop_percent_top/bottom: zum Ränder abschneiden
    # - scale_percent: Kleinere Werte = schneller, aber weniger genau
    # - min_area: Minimale Pixelanzahl für Detection
    # - show_processing: True = zeigt Live-View während Processing
    
    tracker.process_all_videos(
        smooth=True,
        crop_percent_top=0.0,
        crop_percent_bottom=0.0,
        scale_percent=60,
        min_area=500,
        show_processing=False  # Auf True setzen zum Debuggen
    )
    
    # Statistiken ausgeben
    tracker.print_statistics()
    
    # Plots erstellen
    tracker.plot_overlayed_trajectories(save_path='leaf_trajectories.png')
    tracker.plot_with_heatmap(save_path='leaf_heatmap.png')
    
    # Optional: Daten speichern
    tracker.save_trajectories('leaf_trajectories.npz')