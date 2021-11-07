const tglFile=document.querySelector("#toggleTofile");
const tglText=document.querySelector("#toggleTotext");
const formFile=document.querySelector("#file");
const formText=document.querySelector("#text");


function toggle(nt){
    if(nt=="file"){
        tglFile.classList.add("hidden");
        tglText.classList.remove("hidden");
        formText.classList.add("hidden");
        formFile.classList.remove("hidden");
        
    }
    else{
        tglText.classList.add("hidden");
        tglFile.classList.remove("hidden");
        formFile.classList.add("hidden");
        formText.classList.remove("hidden");
    }
}