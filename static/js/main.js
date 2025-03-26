document.addEventListener('DOMContentLoaded', function() {
    const uploadBtn = document.getElementById('uploadBtn');
    const imageInput = document.getElementById('imageInput');
    const uploadMsg = document.getElementById('uploadMsg');
    const uploadedImage = document.getElementById('uploadedImage');
    const optionsSection = document.querySelector('.options-section');
    const processBtn = document.getElementById('processBtn');
    const logDiv = document.getElementById('log');
    const logSection = document.querySelector('.log-section');
    const resultSection = document.querySelector('.result-section');
    let uploadedFilename = '';
  
    uploadBtn.addEventListener('click', function() {
      const file = imageInput.files[0];
      if (!file) {
        alert('Please select an image file.');
        return;
      }
      const formData = new FormData();
      formData.append('image', file);
      
      fetch('/upload', {
        method: 'POST',
        body: formData
      })
      .then(res => res.json())
      .then(data => {
        if (data.error) {
          uploadMsg.textContent = data.error;
        } else {
          uploadMsg.textContent = data.message;
          uploadedFilename = data.filename;
          // Show uploaded image
          uploadedImage.src = `/static/uploads/${uploadedFilename}`;
          uploadedImage.style.display = 'block';
          optionsSection.style.display = 'block';
        }
      })
      .catch(err => console.error(err));
    });
  
    processBtn.addEventListener('click', function() {
      if (!uploadedFilename) {
        alert('Please upload an image first.');
        return;
      }
      // Get selected detail option
      const detailOption = document.querySelector('input[name="detail"]:checked').value;
      
      // Clear previous log and show log section
      logDiv.innerHTML = '';
      logSection.style.display = 'block';
      resultSection.style.display = 'none';
      
      // Start processing and stream logs via EventSource
      const formData = new FormData();
      formData.append('filename', uploadedFilename);
      formData.append('detail_option', detailOption);
  
      // Use fetch with EventSource-like handling (SSE)
      const evtSource = new EventSource('/process?' + new URLSearchParams({
        filename: uploadedFilename,
        detail_option: detailOption
      }));
      
      evtSource.onmessage = function(e) {
        // Try to parse final JSON result if it arrives
        try {
          const result = JSON.parse(e.data);
          // Final result received; display images and master list
          document.getElementById('numberedImg').src = '/' + result.numbered;
          document.getElementById('downloadNumbered').href = '/' + result.numbered;
          document.getElementById('coloredImg').src = '/' + result.colored;
          document.getElementById('downloadColored').href = '/' + result.colored;
          
          // Build master list display
          const masterListDiv = document.getElementById('masterList');
          masterListDiv.innerHTML = '';
          for (const [label, rgb] of Object.entries(result.master_list)) {
            const item = document.createElement('div');
            const colorSquare = document.createElement('div');
            colorSquare.className = 'color-square';
            colorSquare.style.backgroundColor = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
            const text = document.createElement('span');
            text.textContent = ` Region ${label}: rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
            item.appendChild(colorSquare);
            item.appendChild(text);
            masterListDiv.appendChild(item);
          }
          resultSection.style.display = 'block';
          evtSource.close();
        } catch (err) {
          // Otherwise, treat as a log message update
          logDiv.innerHTML += `<p>${e.data}</p>`;
          logDiv.scrollTop = logDiv.scrollHeight;
        }
      };
      
      evtSource.onerror = function() {
        logDiv.innerHTML += `<p>Error during processing.</p>`;
        evtSource.close();
      };
    });
  });
  