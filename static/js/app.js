// Small UI helpers used across pages
document.addEventListener('click', function(e){
  // Accordion toggle
  if(e.target && e.target.matches('.accordion button')){
    const acc = e.target.closest('.accordion');
    acc.classList.toggle('active');
  }
});

// Expose a helper to show transient messages
window.showToast = function(msg, type='info'){
  console.log(type+': '+msg);
}