"use strict";(self.webpackChunk_streamlit_app=self.webpackChunk_streamlit_app||[]).push([[8570],{38570:function(e,r,t){t.d(r,{Z:function(){return A}});var n,o=t(66845),a=t(80318),i="small",s="medium",u="large",l=t(80745),c=t(30067);function d(){return d=Object.assign?Object.assign.bind():function(e){for(var r=1;r<arguments.length;r++){var t=arguments[r];for(var n in t)Object.prototype.hasOwnProperty.call(t,n)&&(e[n]=t[n])}return e},d.apply(this,arguments)}function f(e,r){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);r&&(n=n.filter((function(r){return Object.getOwnPropertyDescriptor(e,r).enumerable}))),t.push.apply(t,n)}return t}function p(e){for(var r=1;r<arguments.length;r++){var t=null!=arguments[r]?arguments[r]:{};r%2?f(Object(t),!0).forEach((function(r){y(e,r,t[r])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):f(Object(t)).forEach((function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(t,r))}))}return e}function y(e,r,t){return r in e?Object.defineProperty(e,r,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[r]=t,e}function b(e){var r;return(r={},y(r,i,"2px"),y(r,s,"4px"),y(r,u,"8px"),r)[e]}var m=(0,l.zo)("div",(function(e){return{width:"100%"}}));m.displayName="StyledRoot",m.displayName="StyledRoot";var g=(0,l.zo)("div",(function(e){var r=e.$theme.sizing;return{display:"flex",marginLeft:r.scale500,marginRight:r.scale500,marginTop:r.scale500,marginBottom:r.scale500}}));g.displayName="StyledBarContainer",g.displayName="StyledBarContainer";var h=(0,l.zo)("div",(function(e){var r=e.$theme,t=e.$size,n=e.$steps,o=r.colors,a=r.sizing,i=r.borders.useRoundedCorners?a.scale0:0;return p({borderTopLeftRadius:i,borderTopRightRadius:i,borderBottomRightRadius:i,borderBottomLeftRadius:i,backgroundColor:(0,c.oo)(o.progressbarTrackFill,"0.16"),height:b(t),flex:1,overflow:"hidden"},n<2?{}:{marginLeft:a.scale300,":first-child":{marginLeft:"0"}})}));h.displayName="StyledBar",h.displayName="StyledBar";var v=(0,l.zo)("div",(function(e){var r=e.$theme,t=e.$value,n=e.$successValue,o=e.$steps,a=e.$index,i=e.$maxValue,s=e.$minValue,u=void 0===s?0:s,l=i||n,c=r.colors,d=r.sizing,f=r.borders,y="".concat(100-100*(t-u)/(l-u),"%"),b="awaits",m="inProgress",g="completed",h="default";if(o>1){var v=(l-u)/o,w=(t-u)/(l-u)*100,P=Math.floor(w/v);h=a<P?g:a===P?m:b}var O=f.useRoundedCorners?d.scale0:0,R={transform:"translateX(-".concat(y,")")},$=h===m?{animationDuration:"2.1s",animationIterationCount:"infinite",animationTimingFunction:r.animation.linearCurve,animationName:{"0%":{transform:"translateX(-102%)",opacity:1},"50%":{transform:"translateX(0%)",opacity:1},"100%":{transform:"translateX(0%)",opacity:0}}}:h===g?{transform:"translateX(0%)"}:{transform:"translateX(-102%)"};return p({borderTopLeftRadius:O,borderTopRightRadius:O,borderBottomRightRadius:O,borderBottomLeftRadius:O,backgroundColor:c.accent,height:"100%",width:"100%",transform:"translateX(-102%)",transition:"transform 0.5s"},o>1?$:R)}));v.displayName="StyledBarProgress",v.displayName="StyledBarProgress";var w=(0,l.zo)("div",(function(e){var r=e.$theme,t=e.$isLeft,n=void 0!==t&&t,o=e.$size,a=void 0===o?s:o,i=r.colors,u=r.sizing,l=r.borders.useRoundedCorners?u.scale0:0,c=b(a),d={display:"inline-block",flex:1,marginLeft:"auto",marginRight:"auto",transitionProperty:"background-position",animationDuration:"1.5s",animationIterationCount:"infinite",animationTimingFunction:r.animation.linearCurve,backgroundSize:"300% auto",backgroundRepeat:"no-repeat",backgroundPositionX:n?"-50%":"150%",backgroundImage:"linear-gradient(".concat(n?"90":"270","deg, transparent 0%, ").concat(i.accent," 25%, ").concat(i.accent," 75%, transparent 100%)"),animationName:n?{"0%":{backgroundPositionX:"-50%"},"33%":{backgroundPositionX:"50%"},"66%":{backgroundPositionX:"50%"},"100%":{backgroundPositionX:"150%"}}:{"0%":{backgroundPositionX:"150%"},"33%":{backgroundPositionX:"50%"},"66%":{backgroundPositionX:"50%"},"100%":{backgroundPositionX:"-50%"}}};return p(p({},n?{borderTopLeftRadius:l,borderBottomLeftRadius:l}:{borderTopRightRadius:l,borderBottomRightRadius:l}),{},{height:c},d)}));w.displayName="StyledInfiniteBar",w.displayName="StyledInfiniteBar";var P=(0,l.zo)("div",(function(e){return p(p({textAlign:"center"},e.$theme.typography.font150),{},{color:e.$theme.colors.contentTertiary})}));P.displayName="StyledLabel",P.displayName="StyledLabel";var O=(y(n={},u,{d:"M47.5 4H71.5529C82.2933 4 91 12.9543 91 24C91 35.0457 82.2933 44 71.5529 44H23.4471C12.7067 44 4 35.0457 4 24C4 12.9543 12.7067 4 23.4471 4H47.5195",width:95,height:48,strokeWidth:8,typography:"LabelLarge"}),y(n,s,{d:"M39 2H60.5833C69.0977 2 76 9.16344 76 18C76 26.8366 69.0977 34 60.5833 34H17.4167C8.90228 34 2 26.8366 2 18C2 9.16344 8.90228 2 17.4167 2H39.0195",width:78,height:36,strokeWidth:4,typography:"LabelMedium"}),y(n,i,{d:"M32 1H51.6271C57.9082 1 63 6.37258 63 13C63 19.6274 57.9082 25 51.6271 25H12.3729C6.09181 25 1 19.6274 1 13C1 6.37258 6.09181 1 12.3729 1H32.0195",width:64,height:26,strokeWidth:2,typography:"LabelSmall"}),n),R=(0,l.zo)("div",(function(e){var r=e.$size,t=e.$inline;return{width:O[r].width+"px",height:O[r].height+"px",position:"relative",display:t?"inline-flex":"flex",alignItems:"center",justifyContent:"center"}}));R.displayName="StyledProgressBarRoundedRoot",R.displayName="StyledProgressBarRoundedRoot";var $=(0,l.zo)("svg",(function(e){var r=e.$size;return{width:O[r].width+"px",height:O[r].height+"px",position:"absolute",fill:"none"}}));$.displayName="_StyledProgressBarRoundedSvg",$.displayName="_StyledProgressBarRoundedSvg";(0,l.Le)($,(function(e){return function(r){return o.createElement(e,d({viewBox:"0 0 ".concat(O[r.$size].width," ").concat(O[r.$size].height),xmlns:"http://www.w3.org/2000/svg"},r))}}));var k=(0,l.zo)("path",(function(e){var r=e.$theme,t=e.$size;return{stroke:r.colors.backgroundTertiary,strokeWidth:O[t].strokeWidth+"px"}}));k.displayName="_StyledProgressBarRoundedTrackBackground",k.displayName="_StyledProgressBarRoundedTrackBackground";(0,l.Le)(k,(function(e){return function(r){return o.createElement(e,d({d:O[r.$size].d},r))}}));var j=(0,l.zo)("path",(function(e){var r=e.$theme,t=e.$size,n=e.$visible,o=e.$pathLength,a=e.$pathProgress;return{visibility:n?"visible":"hidden",stroke:r.colors.borderAccent,strokeWidth:O[t].strokeWidth+"px",strokeDasharray:o,strokeDashoffset:o*(1-a)+""}}));j.displayName="_StyledProgressBarRoundedTrackForeground",j.displayName="_StyledProgressBarRoundedTrackForeground";(0,l.Le)(j,(function(e){return function(r){return o.createElement(e,d({d:O[r.$size].d},r))}}));var S=(0,l.zo)("div",(function(e){var r=e.$theme,t=e.$size;return p({color:r.colors.contentPrimary},r.typography[O[t].typography])}));function z(e){return z="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(e){return typeof e}:function(e){return e&&"function"==typeof Symbol&&e.constructor===Symbol&&e!==Symbol.prototype?"symbol":typeof e},z(e)}S.displayName="StyledProgressBarRoundedText",S.displayName="StyledProgressBarRoundedText";var B=["overrides","getProgressLabel","value","size","steps","successValue","minValue","maxValue","showLabel","infinite","errorMessage","forwardedRef"];function x(){return x=Object.assign?Object.assign.bind():function(e){for(var r=1;r<arguments.length;r++){var t=arguments[r];for(var n in t)Object.prototype.hasOwnProperty.call(t,n)&&(e[n]=t[n])}return e},x.apply(this,arguments)}function L(e,r){return function(e){if(Array.isArray(e))return e}(e)||function(e,r){var t=null==e?null:"undefined"!==typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(null==t)return;var n,o,a=[],i=!0,s=!1;try{for(t=t.call(e);!(i=(n=t.next()).done)&&(a.push(n.value),!r||a.length!==r);i=!0);}catch(u){s=!0,o=u}finally{try{i||null==t.return||t.return()}finally{if(s)throw o}}return a}(e,r)||function(e,r){if(!e)return;if("string"===typeof e)return C(e,r);var t=Object.prototype.toString.call(e).slice(8,-1);"Object"===t&&e.constructor&&(t=e.constructor.name);if("Map"===t||"Set"===t)return Array.from(e);if("Arguments"===t||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t))return C(e,r)}(e,r)||function(){throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}()}function C(e,r){(null==r||r>e.length)&&(r=e.length);for(var t=0,n=new Array(r);t<r;t++)n[t]=e[t];return n}function N(e,r){if(null==e)return{};var t,n,o=function(e,r){if(null==e)return{};var t,n,o={},a=Object.keys(e);for(n=0;n<a.length;n++)t=a[n],r.indexOf(t)>=0||(o[t]=e[t]);return o}(e,r);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(n=0;n<a.length;n++)t=a[n],r.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(o[t]=e[t])}return o}function T(e,r){for(var t=0;t<r.length;t++){var n=r[t];n.enumerable=n.enumerable||!1,n.configurable=!0,"value"in n&&(n.writable=!0),Object.defineProperty(e,n.key,n)}}function E(e,r){return E=Object.setPrototypeOf?Object.setPrototypeOf.bind():function(e,r){return e.__proto__=r,e},E(e,r)}function _(e){var r=function(){if("undefined"===typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"===typeof Proxy)return!0;try{return Boolean.prototype.valueOf.call(Reflect.construct(Boolean,[],(function(){}))),!0}catch(e){return!1}}();return function(){var t,n=X(e);if(r){var o=X(this).constructor;t=Reflect.construct(n,arguments,o)}else t=n.apply(this,arguments);return function(e,r){if(r&&("object"===z(r)||"function"===typeof r))return r;if(void 0!==r)throw new TypeError("Derived constructors may only return object or undefined");return function(e){if(void 0===e)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return e}(e)}(this,t)}}function X(e){return X=Object.setPrototypeOf?Object.getPrototypeOf.bind():function(e){return e.__proto__||Object.getPrototypeOf(e)},X(e)}var V,I,D,M=function(e){!function(e,r){if("function"!==typeof r&&null!==r)throw new TypeError("Super expression must either be null or a function");e.prototype=Object.create(r&&r.prototype,{constructor:{value:e,writable:!0,configurable:!0}}),Object.defineProperty(e,"prototype",{writable:!1}),r&&E(e,r)}(s,e);var r,t,n,i=_(s);function s(){return function(e,r){if(!(e instanceof r))throw new TypeError("Cannot call a class as a function")}(this,s),i.apply(this,arguments)}return r=s,(t=[{key:"componentDidMount",value:function(){}},{key:"render",value:function(){var e=this.props,r=e.overrides,t=void 0===r?{}:r,n=e.getProgressLabel,i=e.value,s=e.size,u=e.steps,l=e.successValue,c=e.minValue,d=e.maxValue,f=e.showLabel,p=e.infinite,y=e.errorMessage,b=e.forwardedRef,O=N(e,B),R=this.props["aria-label"]||this.props.ariaLabel,$=100!==d?d:l,k=L((0,a.jb)(t.Root,m),2),j=k[0],S=k[1],z=L((0,a.jb)(t.BarContainer,g),2),C=z[0],T=z[1],E=L((0,a.jb)(t.Bar,h),2),_=E[0],X=E[1],V=L((0,a.jb)(t.BarProgress,v),2),I=V[0],D=V[1],M=L((0,a.jb)(t.Label,P),2),H=M[0],A=M[1],W=L((0,a.jb)(t.InfiniteBar,w),2),F=W[0],U=W[1],Z={$infinite:p,$size:s,$steps:u,$successValue:$,$minValue:c,$maxValue:$,$value:i};return o.createElement(j,x({ref:b,"data-baseweb":"progress-bar",role:"progressbar","aria-label":R||n(i,$,c),"aria-valuenow":p?null:i,"aria-valuemin":p?null:c,"aria-valuemax":p?null:$,"aria-invalid":!!y||null,"aria-errormessage":y},O,Z,S),o.createElement(C,x({},Z,T),p?o.createElement(o.Fragment,null,o.createElement(F,x({$isLeft:!0,$size:Z.$size},U)),o.createElement(F,x({$size:Z.$size},U))):function(){for(var e=[],r=0;r<u;r++)e.push(o.createElement(_,x({key:r},Z,X),o.createElement(I,x({$index:r},Z,D))));return e}()),f&&o.createElement(H,x({},Z,A),n(i,$,c)))}}])&&T(r.prototype,t),n&&T(r,n),Object.defineProperty(r,"prototype",{writable:!1}),s}(o.Component);D={getProgressLabel:function(e,r,t){return"".concat(Math.round((e-t)/(r-t)*100),"% Loaded")},infinite:!1,overrides:{},showLabel:!1,size:s,steps:1,successValue:100,minValue:0,maxValue:100,value:0},(I="defaultProps")in(V=M)?Object.defineProperty(V,I,{value:D,enumerable:!0,configurable:!0,writable:!0}):V[I]=D;var H=o.forwardRef((function(e,r){return o.createElement(M,x({forwardedRef:r},e))}));H.displayName="ProgressBar";var A=H}}]);